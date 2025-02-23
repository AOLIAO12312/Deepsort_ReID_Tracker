import glob
import os
import cv2
import numpy as np
import torch
from models.yolo.yolo_detector import YoloDetector
from models.deep_sort_pytorch.deepsort_tracker import DeepsortTracker
from src.bounding_box_filter import BoundingBoxFilter
from src.person_database import PersonDatabase
from src.utils import xyxy_to_xywh
from src.utils import get_border

class ReidTracker:
    # Initialize ReidTracker
    def __init__(self,detector_path,deepsort_cfg_path,base_data_path,cfg,reset_queue,device):
        # Initialize yolo detector
        print("Loading object detector...")
        self.detector = YoloDetector(detector_path, device)

        # Initialize deepsortTracker
        print("Loading deepsort tracker...")
        deepsort_tracker = DeepsortTracker(deepsort_cfg_path, device)
        self.tracker = deepsort_tracker.get_tracker()

        # Initialize person feature database

        self.person_database = PersonDatabase(cfg)

        # Record of the index of the frame
        self.frame_idx = 0
        # Deepsort ID to Person's Mapping table
        self.deepsort_to_athlete = {}
        # Deepsort id lost count if the count is too high, id will be removed
        self.id_lost_count = {}
        self.LOST_THRESHOLD = 15

        # Queue for receiving external input that requires to reset personnel
        self.reset_queue = reset_queue

        self.bounding_box_filter = None
        self.base_data_path = base_data_path
        self.cfg = cfg

        self.block_id = {}
        self.person_conf = {}
        self.name_seq = 1
        print("Loading base data...")
        self.load_base_data()
        self.matrix = None


    def load_base_data(self):
        """
        Loads base data from the specified directory, reads image files from each folder,
        and adds the images to the person database.

        The method iterates through all folders in the base data path, reads images with
        ".png" extension from each folder, and adds them to the database under the corresponding person's name.
        """
        for folder_name in os.listdir(self.base_data_path):
            folder_path = os.path.join(self.base_data_path, folder_name)
            if os.path.isdir(folder_path):
                image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
                images = [cv2.imread(img_path) for img_path in image_paths]
                self.person_database.add_person(folder_name, images)
                print(f"Added person: {folder_name} with {len(images)} images.")

    def identify(self,cropped_images:list):
        """
        Receive cropped images list need to be identified and return candidates results and L2 distance

        Parameters
        ----------
        cropped_images:person's image need to be identified

        Returns
        -------
        data: L2 distance between input image and database reference image
        """
        data = []
        for cropped_image in cropped_images:
            results = self.person_database.search(cropped_image, 3)
            data.append(results)
        return data

    def map_deepsort_to_athlete(self,tracking_results, orig_img):
        """
        Map the DeepSORT ID to the actual athlete identity ID

        Parameters
        ----------
        tracking_results: list
            deepsort tracking outputs
        orig_img: numpy.ndarray
            original image of this frame

        Returns
        -------
        mapped_results: list
            final mapped results of each bbox
        """
        mapped_results = []
        unassigned_tracks = []
        unassigned_deepsort_ids = []
        update_tracks_names = []
        update_tracks_image = []
        active_ids = set()
        for track in tracking_results:
            deepsort_id = track[4]
            bbox = track[:4]
            active_ids.add(deepsort_id)
            if deepsort_id not in self.deepsort_to_athlete:
                x1, y1, x2, y2 = map(int, bbox)
                cropped_image = orig_img[y1:y2, x1:x2]
                unassigned_tracks.append(cropped_image)
                unassigned_deepsort_ids.append(deepsort_id)
            else:
                if self.frame_idx % 30 == 0:
                    x1, y1, x2, y2 = map(int, bbox)
                    cropped_image = orig_img[y1:y2, x1:x2]
                    update_tracks_image.append(cropped_image)
                    update_tracks_names.append(self.deepsort_to_athlete[deepsort_id])

        # Dynamically write the person's features to the database every certain number of frames and rebuild the index
        if self.frame_idx % 30 == 0:
            if not self.reset_queue.empty():
                user_input = self.reset_queue.get()
                print(f"\nWaiting for {user_input} information to be reset...")
                self.person_database.update_person_feature_and_rebuild_index(update_tracks_names, update_tracks_image, 3,
                                                                        [user_input])
                existing_deepsort_ids = [k for k, v in self.deepsort_to_athlete.items() if v == user_input]
                if len(existing_deepsort_ids) > 0:
                    del self.deepsort_to_athlete[existing_deepsort_ids[0]]
                    self.block_id[existing_deepsort_ids[0]] = user_input
            else:
                self.person_database.update_person_feature_and_rebuild_index(update_tracks_names, update_tracks_image, 3,
                                                                        [])

        datas = self.identify(unassigned_tracks)
        # handle the issue of multiple DeepSORT IDs corresponding to the same athlete.
        for idx, (deepsort_id,data) in enumerate(zip(unassigned_deepsort_ids,datas)):
            for (candidate,distance) in data:
                if (not deepsort_id in self.block_id) or (self.block_id[deepsort_id] != candidate):
                    # if the L2 distance is below the threshold, it can be considered that the identification confidence is high enough to assign an ID.
                    if distance < 0.37:
                        if candidate in self.deepsort_to_athlete.values():
                            continue
                        else:
                            self.deepsort_to_athlete[deepsort_id] = candidate
                            break
                    else:
                        break
                else:
                    continue

        # Add the result to the final mapping
        for track in tracking_results:
            deepsort_id = track[4]
            if deepsort_id in self.deepsort_to_athlete:
                mapped_results.append([*track[:4], self.deepsort_to_athlete[deepsort_id]])
            else:
                mapped_results.append([*track[:4], f'Unknown {deepsort_id}'])
        self.handle_lost_ids(active_ids)  # handle lost id and delete it lost for a long time
        return mapped_results

    def multi_frame_map_deepsort_to_athlete(self, tracking_resultses, orig_imgs):
        deepsort_id_to_images = {}
        mapped_resultses = []
        for idx,(tracking_results,orig_img) in enumerate(zip(tracking_resultses,orig_imgs)):
            for tracking_result in tracking_results:
                deepsort_id = tracking_result[4]
                if deepsort_id in self.deepsort_to_athlete: # Skip when person was mapped
                    continue
                bbox = tracking_result[:4]
                x1, y1, x2, y2 = map(int, bbox)
                cropped_image = orig_img[y1:y2, x1:x2]
                if deepsort_id in deepsort_id_to_images:
                    deepsort_id_to_images[deepsort_id].append(cropped_image)
                else:
                    deepsort_id_to_images[deepsort_id] = [cropped_image]

        # Assume 15 frames as a group
        # if tracking id appear less than half frames, discard it
        update_names = []
        update_images = []
        for i,(deepsort_id,cropped_images) in enumerate(deepsort_id_to_images.items()):
            if len(cropped_images) < int(len(cropped_images)/2):
                continue
            sliced_images = cropped_images[::self.cfg['reid_tracker']['sample_density']]
            # cv2.imshow("identify",sliced_images[0])
            # cv2.waitKey(0)
            # cv2.destroyWindow("identify")
            results = self.person_database.multi_frame_search(sliced_images,4)
            for result in results:
                if deepsort_id in self.block_id:
                    if self.block_id[deepsort_id] == result[0]:
                        continue
                if result[1] > self.cfg['reid_tracker']['min_conf']:
                    if result[0] in self.deepsort_to_athlete.values():
                        existing_deepsort_id = -1
                        for existing_deepsort_id, person in list(self.deepsort_to_athlete.items()):
                            if person == result[0]:
                                break
                        if deepsort_id != -1:
                            exist_frame_count = 0
                            for tracking_results in tracking_resultses:
                                for line in tracking_results:
                                    if line[4] == existing_deepsort_id:
                                        exist_frame_count += 1
                                        break
                                    else:
                                        continue
                            if exist_frame_count < len(orig_imgs):
                                self.person_conf[result[0]] -= 0.1
                        if result[1] > self.person_conf[result[0]]:
                            for existing_deepsort_id, person in list(self.deepsort_to_athlete.items()):
                                if person == result[0]:
                                    del self.deepsort_to_athlete[existing_deepsort_id]
                                    break
                            self.deepsort_to_athlete[deepsort_id] = result[0]
                            self.person_conf[result[0]] = result[1]
                            break
                        continue
                    else:
                        self.deepsort_to_athlete[deepsort_id] = result[0]
                        self.person_conf[result[0]] = result[1]
                        break
                else:
                    # if self.name_seq < 10:
                    #     self.person_database.add_person(f"Person_{self.name_seq}",sliced_images)
                    #     self.deepsort_to_athlete[deepsort_id] = f"Person_{self.name_seq}"
                    #     self.person_conf[f"Person_{self.name_seq}"] = 1
                    #     self.name_seq += 1
                    break

        tracking_results = tracking_resultses[int(len(tracking_resultses)/2)]
        orig_img = orig_imgs[int(len(orig_imgs)/2)]
        for track in tracking_results:
            bbox = track[:4]
            deepsort_id = track[4]
            if deepsort_id in self.deepsort_to_athlete:
                x1,y1,x2,y2 = map(int,bbox)
                cropped_image = orig_img[y1:y2, x1:x2]
                name = self.deepsort_to_athlete[deepsort_id]
                update_names.append(name)
                update_images.append(cropped_image)

        if not self.reset_queue.empty():
            user_input = self.reset_queue.get()
            print(f"\nWaiting for {user_input} information to be reset...")
            self.person_database.update_person_feature(update_names,update_images,self.cfg['reid_tracker']['reinforce_tensity'],[user_input])
            existing_deepsort_ids = [k for k, v in self.deepsort_to_athlete.items() if v == user_input]
            if len(existing_deepsort_ids) > 0:
                del self.deepsort_to_athlete[existing_deepsort_ids[0]]
                self.block_id[existing_deepsort_ids[0]] = user_input
        else:
            self.person_database.update_person_feature(update_names,update_images,self.cfg['reid_tracker']['reinforce_tensity'],[])

        for idx,tracking_results in enumerate(tracking_resultses):
            mapped_results = []
            for tracking_result in tracking_results:
                bbox = tracking_result[:4]
                deepsort_id = tracking_result[4]
                if deepsort_id in self.deepsort_to_athlete:
                    mapped_results.append([*bbox,self.deepsort_to_athlete[deepsort_id]])
                else:
                    mapped_results.append([*bbox, f'Unknown {deepsort_id}'])
            mapped_resultses.append(mapped_results)
        return mapped_resultses


    def handle_lost_ids(self,active_ids):
        """
        Handles the case when certain IDs are lost during tracking by updating the
        lost count and removing those IDs if they exceed the threshold.

        Parameters
        ----------
        active_ids : set
            A set of currently active IDs from DeepSORT.

        Returns
        -------
        None
        """
        lost_ids = set(self.deepsort_to_athlete.keys()) - active_ids
        for lost_id in lost_ids:
            self.id_lost_count[lost_id] = self.id_lost_count.get(lost_id, 0) + 1
            if self.id_lost_count[lost_id] >= self.LOST_THRESHOLD:
                del self.deepsort_to_athlete[lost_id]
                del self.id_lost_count[lost_id]


    def update(self,frame):
        """
        Processes a given series frame, detects objects, filters bounding boxes, and performs continuous tracking
        on the detected individuals. If valid detections are found, it updates the tracker and maps the results
        to the corresponding athletes.

        Parameters
        ----------
        frame : numpy.ndarray
            The current video frame to be processed.

        Returns
        -------
        mapped_results : list
            A list of the results where each result is mapped to a specific athlete based on the tracker output.
            If no valid frame or detections are found, an empty list is returned.
        """
        if frame is not None:
            if self.bounding_box_filter is None:
                bound = get_border(frame.copy())
                self.bounding_box_filter = BoundingBoxFilter(bound,0.1,0.4)
                return []
            tracker_outputs = []
            result = self.detector.get_result(frame)[0]
            frame, xyxy, conf = self.bounding_box_filter.box_filter(frame, result)
            if xyxy is not None:
                xywhs = torch.empty(0, 4)
                confess = torch.empty(0, 1)
                for i, (bbox, confidence) in enumerate(zip(xyxy, conf)):
                    x1, y1, x2, y2 = map(int, bbox)
                    x_c, y_c, w, h = xyxy_to_xywh(x1, y1, x2, y2)
                    xywhs = torch.cat((xywhs, torch.tensor([x_c, y_c, w, h]).unsqueeze(0)), dim=0)
                    confess = torch.cat((confess, torch.tensor([confidence]).unsqueeze(0)), dim=0)
                # Perform continuous human tracking
                tracker_outputs = self.tracker.update(xywhs, confess, frame)
            else:
                self.tracker.increment_ages()
            # Map DeepSORT results to specific athlete identities
            mapped_results = self.map_deepsort_to_athlete(tracker_outputs, frame)
            self.frame_idx += 1
            return mapped_results
        else:
            return []

    def get_matrix(self):
        return self.matrix

    def multi_frame_update(self, frames):
        """
        Processes a given series frame, detects objects, filters bounding boxes, and performs continuous tracking
        on the detected individuals. If valid detections are found, it updates the tracker and maps the results
        to the corresponding athletes.

        Parameters
        ----------
        frames : list
            The current video frame to be processed.

        Returns
        -------
        mapped_results : list
            A list of the results where each result is mapped to a specific athlete based on the tracker output.
            If no valid frame or detections are found, an empty list is returned.
        """
        if frames is not None:
            results = self.detector.get_result(frames)
            tracking_resultses = []
            for idx,frame in enumerate(frames):
                if frame is not None:
                    if self.bounding_box_filter is None:
                        bound = get_border(frame.copy())
                        self.bounding_box_filter = BoundingBoxFilter(bound, 0.1, 0.4)
                        width, height = 650, 500
                        pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                                           dtype='float32')
                        pts_src = np.array(bound, dtype='float32')
                        self.matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
                frame,xyxy,conf = self.bounding_box_filter.box_filter(frame,results[idx])
                tracking_results = []
                if xyxy is not None:
                    xywhs = torch.empty(0, 4)
                    confess = torch.empty(0, 1)
                    for i, (bbox, confidence) in enumerate(zip(xyxy, conf)):
                        x1, y1, x2, y2 = map(int, bbox)
                        x_c, y_c, w, h = xyxy_to_xywh(x1, y1, x2, y2)
                        xywhs = torch.cat((xywhs, torch.tensor([x_c, y_c, w, h]).unsqueeze(0)), dim=0)
                        confess = torch.cat((confess, torch.tensor([confidence]).unsqueeze(0)), dim=0)
                    # Perform continuous human tracking
                    tracking_results = self.tracker.update(xywhs, confess, frame)
                else:
                    self.tracker.increment_ages()
                tracking_resultses.append(tracking_results)
            mapped_resultses = self.multi_frame_map_deepsort_to_athlete(tracking_resultses,frames)
            return mapped_resultses
        else:
            return []


        #     if self.bounding_box_filter is None:
        #         bound = get_border(frame.copy())
        #         self.bounding_box_filter = BoundingBoxFilter(bound, 0.1, 0.4)
        #         return []
        #     tracker_outputs = []
        #     result = self.detector.get_result(frame)
        #     frame, xyxy, conf = self.bounding_box_filter.box_filter(frame, result)
        #     if xyxy is not None:
        #         xywhs = torch.empty(0, 4)
        #         confess = torch.empty(0, 1)
        #         for i, (bbox, confidence) in enumerate(zip(xyxy, conf)):
        #             x1, y1, x2, y2 = map(int, bbox)
        #             x_c, y_c, w, h = xyxy_to_xywh(x1, y1, x2, y2)
        #             xywhs = torch.cat((xywhs, torch.tensor([x_c, y_c, w, h]).unsqueeze(0)), dim=0)
        #             confess = torch.cat((confess, torch.tensor([confidence]).unsqueeze(0)), dim=0)
        #         # Perform continuous human tracking
        #         tracker_outputs = self.tracker.update(xywhs, confess, frame)
        #     else:
        #         self.tracker.increment_ages()
        #     # Map DeepSORT results to specific athlete identities
        #     mapped_results = self.map_deepsort_to_athlete(tracker_outputs, frame)
        #     self.frame_idx += 1
        #     return mapped_results
        # else:
        #     return []