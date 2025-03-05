import threading

import numpy as np

from models.yolo.yolo_detector import YoloDetector
import cv2
import queue
from src.reid_tracker import ReidTracker
import yaml
from src.bird_eye_view import BirdEyeView
from src.utils import draw_reid_tracking_results

reset_queue = queue.Queue()

def load_config(config_path:str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def wait_for_input():
    while True:
        user_input = input("Please enter the person needs to be reset:")
        reset_queue.put(user_input)

config = load_config("E:\Deepsort_ReID_Tracker\configs\config.yaml")

camera1_results_queue = queue.Queue()
camera2_results_queue = queue.Queue()
camera3_results_queue = queue.Queue()
camera4_results_queue = queue.Queue()

def read_frame_and_track(video_capture,reid_tracker,camera_results_queue):
    ret = True
    frames = []
    for i in range(15):
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    if not ret:
        print("Read Frame Finish")
        exit(0)
    tracking_resultses = reid_tracker.multi_frame_update(frames)
    camera_results_queue.put((frames,tracking_resultses))

def conbine_field_and_frame(field,frame,tracking_results):
    draw_reid_tracking_results(tracking_results, frame)
    scale_factor = 0.55
    field_resized = cv2.resize(field, (0, 0), fx=scale_factor, fy=scale_factor)
    fh, fw, _ = frame.shape
    th, tw, _ = field_resized.shape
    x_offset = 10
    y_offset = fh - th - 10
    roi = frame[y_offset:y_offset + th, x_offset:x_offset + tw]
    alpha = 0.8
    blended = cv2.addWeighted(field_resized, alpha, roi, 1 - alpha, 0)
    frame[y_offset:y_offset + th, x_offset:x_offset + tw] = blended
    return frame


def compute_iou(box1, box2):
    """计算两个边界框的 IoU（Intersection over Union）"""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y4)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0, area1, area2  # 额外返回框面积


def check_contact(tracking_results_group, iou_threshold=0.1, score_threshold=2000):
    """
    计算球员的加权接触评分，若总分超过 `score_threshold`，则判定为关键帧。
    """
    team_A = {"1", "2", "3", "4"}
    team_B = {"5", "6", "7", "8"}

    contact_score = 0  # 记录总接触评分

    for camera_idx,tracking_results in enumerate(tracking_results_group):
        all_detections = list(tracking_results)

        for i in range(len(all_detections)):
            box1, id1 = all_detections[i][:4], str(all_detections[i][4])

            for j in range(i + 1, len(all_detections)):
                box2, id2 = all_detections[j][:4], str(all_detections[j][4])

                if (id1 in team_A and id2 in team_B) or (id2 in team_A and id1 in team_B):
                    iou, area1, area2 = compute_iou(box1, box2)
                    if iou > iou_threshold:
                        weight = min(area1, area2)  # 取较小目标框
                        contact_score += iou * weight  # 关键修改
                        print(f"camera{camera_idx + 1},id1 = {id1},id2 = {id2},weight = {weight},iou = {iou},score = {contact_score},frame_idx = {frame_idx}")
                        if contact_score >= score_threshold:
                            return True  # 立即返回关键帧

    return False  # 低于阈值，不是关键帧


if __name__ == '__main__':

    detector_1 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_2 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_3 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_4 = YoloDetector(config['yolo']['model_path'], 'cuda:0')

    bird_view_1 = BirdEyeView()
    bird_view_2 = BirdEyeView()
    bird_view_3 = BirdEyeView()
    bird_view_4 = BirdEyeView()

    reid_tracker_1 = ReidTracker(detector_1,config['deepsort']['deepsort_cfg_path'],
                                                'E:\Deepsort_ReID_Tracker\data\input\Camera1_base_data',config, reset_queue,"camera1" ,'cuda:0')
    reid_tracker_2 = ReidTracker(detector_2, config['deepsort']['deepsort_cfg_path'],
                                 'E:\Deepsort_ReID_Tracker\data\input\Camera2_base_data', config, reset_queue, "camera2",'cuda:0')
    reid_tracker_3 = ReidTracker(detector_3, config['deepsort']['deepsort_cfg_path'],
                                 'E:\Deepsort_ReID_Tracker\data\input\Camera3_base_data', config, reset_queue,"camera3" ,'cuda:0')
    reid_tracker_4 = ReidTracker(detector_4, config['deepsort']['deepsort_cfg_path'],
                                 'E:\Deepsort_ReID_Tracker\data\input\Camera4_base_data', config, reset_queue, "camera4",'cuda:0')

    video_capture_1 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\short_version\Camera1.mp4')
    video_capture_2 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\short_version\Camera2.mp4')
    video_capture_3 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\short_version\Camera3.mp4')
    video_capture_4 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\short_version\Camera4.mp4')


    # total_frames = int(video_capture_1.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_width = int(video_capture_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video_capture_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # output_video_path = 'E:/Deepsort_ReID_Tracker/data/output/tracked_video_GRID.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer_1 = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))
    # video_writer_2 = cv2.VideoWriter(output_video_path + str(2), fourcc, 30, (frame_width, frame_height))
    # video_writer_3 = cv2.VideoWriter(output_video_path + str(3), fourcc, 30, (frame_width, frame_height))
    # video_writer_4 = cv2.VideoWriter(output_video_path + str(4), fourcc, 30, (frame_width, frame_height))


    # input_thread = threading.Thread(target=wait_for_input, daemon=True)
    # input_thread.start()

    frame_idx = 0
    output_dir = f"E:/Deepsort_ReID_Tracker/data/output/keyframe/"

    while True:

        track_thread_1 = threading.Thread(target=read_frame_and_track,
                                          args=(video_capture_1,reid_tracker_1,camera1_results_queue))
        track_thread_2 = threading.Thread(target=read_frame_and_track,
                                          args=(video_capture_2, reid_tracker_2, camera2_results_queue))
        track_thread_3 = threading.Thread(target=read_frame_and_track,
                                          args=(video_capture_3, reid_tracker_3, camera3_results_queue))
        track_thread_4 = threading.Thread(target=read_frame_and_track,
                                          args=(video_capture_4, reid_tracker_4, camera4_results_queue))
        track_thread_1.start()
        track_thread_2.start()
        track_thread_3.start()
        track_thread_4.start()

        track_thread_1.join()
        track_thread_2.join()
        track_thread_3.join()
        track_thread_4.join()

        frames_1,tracking_resultses_1 = camera1_results_queue.get()
        frames_2,tracking_resultses_2 = camera2_results_queue.get()
        frames_3,tracking_resultses_3 = camera3_results_queue.get()
        frames_4,tracking_resultses_4 = camera4_results_queue.get()

        if bird_view_1.matrix is None:
            bird_view_1.matrix = reid_tracker_1.matrix
        if bird_view_2.matrix is None:
            bird_view_2.matrix = reid_tracker_2.matrix
        if bird_view_3.matrix is None:
            bird_view_3.matrix = reid_tracker_3.matrix
        if bird_view_4.matrix is None:
            bird_view_4.matrix = reid_tracker_4.matrix

        for idx, (tracking_results_1,tracking_results_2,tracking_results_3,tracking_results_4) in enumerate(zip(tracking_resultses_1,tracking_resultses_2,tracking_resultses_3,tracking_resultses_4)):
            field1 = bird_view_1.draw_bird_view(tracking_results_1)
            field2 = bird_view_2.draw_bird_view(tracking_results_2)
            field3 = bird_view_3.draw_bird_view(tracking_results_3)
            field4 = bird_view_4.draw_bird_view(tracking_results_4)

            monitor_1 = conbine_field_and_frame(field1, frames_1[idx].copy(), tracking_results_1)
            monitor_2 = conbine_field_and_frame(field2, frames_2[idx].copy(), tracking_results_2)
            monitor_3 = conbine_field_and_frame(field3, frames_3[idx].copy(), tracking_results_3)
            monitor_4 = conbine_field_and_frame(field4, frames_4[idx].copy(), tracking_results_4)

            group = [tracking_results_1,tracking_results_2,tracking_results_3,tracking_results_4]
            ret = check_contact(group)
            if ret is True: # 接触判定代码段

                print(f"Keyframe = {frame_idx} extracted")
                filename = f"{output_dir}camera1/collision_frame_idx={frame_idx}.jpg"
                cv2.imwrite(filename, frames_1[idx])
                filename = f"{output_dir}camera2/collision_frame_idx={frame_idx}.jpg"
                cv2.imwrite(filename, frames_2[idx])
                filename = f"{output_dir}camera3/collision_frame_idx={frame_idx}.jpg"
                cv2.imwrite(filename, frames_3[idx])
                filename = f"{output_dir}camera4/collision_frame_idx={frame_idx}.jpg"
                cv2.imwrite(filename, frames_4[idx])

            # 2 * 2 concat
            h1 = np.hstack((monitor_1, monitor_2))
            h2 = np.hstack((monitor_3, monitor_4))
            grid = np.vstack((h1, h2))
            # cv2.imshow("Monitor1", monitor_1)
            # cv2.imshow("Monitor2", monitor_2)
            # cv2.imshow("Monitor3", monitor_3)
            # cv2.imshow("Monitor4", monitor_4)
            scale = 0.5
            grid_resized = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow("Monitor Grid", grid_resized)
            cv2.waitKey(1)
            # video_writer_1.write(grid_resized)
            frame_idx += 1

    # ret = True
    # while True:
    #     frames = []
    #     for i in range(15):
    #         ret, frame = video_capture.read()
    #         if not ret:
    #             break
    #         frames.append(frame)
    #     if not ret:
    #         break
    #
    #     tracking_resultses = reid_tracker.multi_frame_update(frames)
    #
    #     if bird_view.matrix is None:
    #         bird_view.matrix = reid_tracker.matrix
    #
    #     for idx,(tracking_results,frame) in enumerate(zip(tracking_resultses,frames)):
    #         draw_reid_tracking_results(tracking_results, frame)
    #         field = bird_view.draw_bird_view(tracking_results)
    #         scale_factor = 0.55
    #         field_resized = cv2.resize(field, (0, 0), fx=scale_factor, fy=scale_factor)
    #         fh, fw, _ = frame.shape
    #         th, tw, _ = field_resized.shape
    #         x_offset = 10
    #         y_offset = fh - th - 10
    #         roi = frame[y_offset:y_offset + th, x_offset:x_offset + tw]
    #         alpha = 0.8
    #         blended = cv2.addWeighted(field_resized, alpha, roi, 1 - alpha, 0)
    #         frame[y_offset:y_offset + th, x_offset:x_offset + tw] = blended
    #
    #         cv2.imshow("Monitor", frame)
    #         video_writer.write(frame)
    #         cv2.waitKey(1)
    #
    # video_capture.release()
    # video_writer.release()
    # cv2.destroyAllWindows()
    # print(f"Processing complete, video has been saved as: {output_video_path}")