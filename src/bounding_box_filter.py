import cv2
import numpy as np
import torch
from src.utils import yolo2coordinates


class BoundingBoxFilter:

    def __init__(self,bound,conf,limit_percentage):
        self.bound = bound
        self.conf = conf
        self.limit_percentage = limit_percentage


    def box_filter(self,frame,result):
        frame = frame.copy()
        orig_img = frame
        detections = result.boxes.xywhn.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        person_detections = []
        person_confidences = []
        for i, (detection,confidence,class_id) in enumerate(zip(detections,confidences,class_ids)):
            x1, y1, x2, y2 = yolo2coordinates(detection[0],detection[1],detection[2],detection[3],len(orig_img[0]),len(orig_img))
            if class_id == 0 and confidence >= self.conf:
                person_detections.append((x1,y1,x2,y2))
                person_confidences.append(confidence)
        pt1 = self.bound[0]
        pt2 = self.bound[1]
        pt3 = self.bound[2]
        pt4 = self.bound[3]

        color = (68, 70, 254)
        thickness = 2

        cv2.line(frame, pt1, pt2, color, thickness)
        cv2.line(frame, pt2, pt3, color, thickness)
        cv2.line(frame, pt3, pt4, color, thickness)
        cv2.line(frame, pt1, pt4, color, thickness)
        image_with_line = frame.copy()
        image_rgb = cv2.cvtColor(image_with_line, cv2.COLOR_BGR2RGB)

        target_color = (254, 70, 68)  # RGB(254, 70, 68)


        lower_bound = np.array([target_color[0] - 10, target_color[1] - 10, target_color[2] - 10])
        upper_bound = np.array([target_color[0] + 10, target_color[1] + 10, target_color[2] + 10])

        mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            xyxy = torch.empty(0, 4)
            conf = torch.empty(0)
            if len(approx) == 4:
                cv2.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

                mask_roi_bound = np.zeros_like(mask)
                cv2.fillPoly(mask_roi_bound, [approx], 255)


                for i, (detection,person_confidence) in enumerate(zip(person_detections,person_confidences)):
                    x1, y1, x2, y2 = map(int, detection)

                    bottom_edge_roi = mask_roi_bound[y2 - 2:y2, x1:x2]

                    bottom_edge_masked = np.sum(bottom_edge_roi == 255)
                    total_bottom_edge_area = bottom_edge_roi.size

                    mask_percentage = bottom_edge_masked / total_bottom_edge_area

                    if mask_percentage > self.limit_percentage:
                        xyxy = torch.cat((xyxy, torch.tensor([x1, y1, x2, y2]).unsqueeze(0)), dim=0)
                        conf = torch.cat((conf,torch.tensor([person_confidence])),dim = 0)

            return orig_img,xyxy,conf