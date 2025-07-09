import cv2
import torch
from tqdm import tqdm

from models.deep_sort_pytorch.deepsort_tracker import DeepsortTracker
from models.yolo.yolo_detector import YoloDetector
from src.bounding_box_filter import BoundingBoxFilter
from src.utils import xyxy_to_xywh, draw_reid_tracking_results, get_border

if __name__ == '__main__':
    deepsort_tracker = DeepsortTracker("E:/Deepsort_ReID_Tracker/configs/deep_sort.yaml", 'cuda:0')
    tracker = deepsort_tracker.get_tracker()
    detector = YoloDetector("E:/Deepsort_ReID_Tracker/models/yolo/weights/yolo11n.pt", 'cuda:0')
    bounds = {'camera1': [(126, 287), (700, 112), (1118, 192), (726, 704)],
              'camera2': [(494, 719), (244, 160), (632, 93), (1189, 273)],
              'camera3': [(1130, 158), (760, 686), (140, 249), (709, 64)],
              'camera4': [(535, 150), (1089, 277), (587, 718), (148, 244)]}


    video_capture_3 = cv2.VideoCapture('E:/Deepsort_ReID_Tracker/data/input/long_video/Camera3.mp4')
    total_frames = int(video_capture_3.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture_3.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture_3.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = 'E:/Deepsort_ReID_Tracker/data/output/ablation_output/deepsort_no_filter_test.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer2 = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

    bounding_box_filter = BoundingBoxFilter([(1130, 158), (760, 686), (140, 249), (709, 64)], 0.1, 0.4)


    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while True:
            ret, frame = video_capture_3.read()
            if not ret:
                print("Finished")
                break

            result = detector.get_result(frame)[0]
            frame, xyxy, conf = bounding_box_filter.box_filter(frame, result)
            tracker_outputs = []
            if xyxy is not None:
                xywhs = torch.empty(0, 4)
                confess = torch.empty(0, 1)
                for i, (bbox, confidence) in enumerate(zip(xyxy, conf)):
                    x1, y1, x2, y2 = map(int, bbox)
                    x_c, y_c, w, h = xyxy_to_xywh(x1, y1, x2, y2)
                    xywhs = torch.cat((xywhs, torch.tensor([x_c, y_c, w, h]).unsqueeze(0)), dim=0)
                    confess = torch.cat((confess, torch.tensor([confidence]).unsqueeze(0)), dim=0)
                tracker_outputs = tracker.update(xywhs, confess, frame)
            else:
                tracker.increment_ages()

            draw_reid_tracking_results(tracker_outputs, frame)
            # cv2.imshow("Deepsort Monitor", frame)
            video_writer2.write(frame)
            # cv2.waitKey(1)
            pbar.update(1)  # 更新进度条

    video_capture_3.release()
    video_writer2.release()
    # cv2.destroyAllWindows()
