import threading
import numpy as np
from models.yolo.yolo_detector import YoloDetector
import cv2
import queue
import os

from src.camera_fusion import CameraFusion
from src.reid_tracker import ReidTracker
import yaml
from src.bird_eye_view import BirdEyeView
from src.utils import draw_reid_tracking_results

def load_config(config_path:str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def conbine_field_and_frame(field,frame,tracking_results,bound):
    pt1 = bound[0]
    pt2 = bound[1]
    pt3 = bound[2]
    pt4 = bound[3]

    color = (68, 70, 254)
    thickness = 2

    cv2.line(frame, pt1, pt2, color, thickness)
    cv2.line(frame, pt2, pt3, color, thickness)
    cv2.line(frame, pt3, pt4, color, thickness)
    cv2.line(frame, pt1, pt4, color, thickness)
    draw_reid_tracking_results(tracking_results, frame)
    scale_factor = 0.55
    # field_resized = cv2.resize(field, (0, 0), fx=scale_factor, fy=scale_factor)
    # fh, fw, _ = frame.shape
    # # th, tw, _ = field_resized.shape
    # x_offset = 10
    # y_offset = fh - th - 10
    # roi = frame[y_offset:y_offset + th, x_offset:x_offset + tw]
    # alpha = 0.8
    # blended = cv2.addWeighted(field_resized, alpha, roi, 1 - alpha, 0)
    # frame[y_offset:y_offset + th, x_offset:x_offset + tw] = blended
    return frame

config = load_config("/configs/config.yaml")

reset_queue = queue.Queue()
fix_queue = queue.Queue()

if __name__ == '__main__':
    detector = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    bird_view = BirdEyeView()

    reid_tracker = ReidTracker(detector,config['deepsort']['deepsort_cfg_path'],
                                                None,config, reset_queue,fix_queue,"camera0",'cuda:0')

    frame_path = "F:\\大创资料\\MOT17\\MOT17\\test\\MOT17-08-DPM\\img1"

    image_files = sorted([f for f in os.listdir(frame_path) if f.endswith(".jpg")],
                         key=lambda x: int(os.path.splitext(x)[0]))
    frame_idx = 0

    i = 0
    frames = []
    for img_file in image_files:
        img_path = os.path.join(frame_path, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Failed to load {img_path}")
            continue
        if i < 15:
            frames.append(frame)
            i += 1
        else:
            tracking_resultses = reid_tracker.multi_frame_update(frames)

            if frame_idx <= 15:
                bound = reid_tracker.bounding_box_filter.bound

            for idx,(tracking_results) in enumerate(zip(tracking_resultses)):
                eye_position = []
                field = bird_view.draw_bird_view(tracking_results, eye_position)
                monitor = conbine_field_and_frame(field, frames[idx].copy(), tracking_results, bound)
                cv2.imshow("Monitor", monitor)
                cv2.waitKey(1)

            frames.clear()
            frames.append(frame)
            i = 0
        frame_idx += 1


