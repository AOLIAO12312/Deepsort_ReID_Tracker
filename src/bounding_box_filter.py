import cv2
import numpy as np
import torch
from src.utils import yolo2coordinates


class BoundingBoxFilter:
    def __init__(self, bound, conf, limit_percentage):
        self.bound = bound
        self.conf = conf
        self.limit_percentage = limit_percentage

    def parse_dets_to_yolov8_format(self,dets, orig_dim_list):
        """
        将 YOLOv3 风格的 dets 转换为 YOLOv8 风格的结果。
        输入：
            dets: torch.Tensor, shape=[n, 8], 格式：[image_idx, x1, y1, x2, y2, obj_score, class_score, class_idx]
            orig_dim_list: torch.Tensor, shape=[batch_size, 4], 每行格式：[w, h, w, h]
        输出：
            detections: np.ndarray, shape=[n, 4], xywhn 格式（归一化中心点+宽高）
            confidences: np.ndarray, shape=[n]
            class_ids: np.ndarray, shape=[n]
        """
        dets_np = dets.cpu().numpy()

        # 拆分字段
        image_idxs = dets_np[:, 0].astype(int)
        x1 = dets_np[:, 1]
        y1 = dets_np[:, 2]
        x2 = dets_np[:, 3]
        y2 = dets_np[:, 4]
        obj_scores = dets_np[:, 5]
        class_scores = dets_np[:, 6]
        class_ids = dets_np[:, 7]

        # 计算宽高和中心点（未归一化）
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        # 归一化坐标（除以原图宽高）
        orig_dims = orig_dim_list[image_idxs][:, :2].cpu().numpy()  # 取每张图原始尺寸 [w, h]
        norm_cx = cx / orig_dims[:, 0]
        norm_cy = cy / orig_dims[:, 1]
        norm_w = w / orig_dims[:, 0]
        norm_h = h / orig_dims[:, 1]

        detections = np.stack([norm_cx, norm_cy, norm_w, norm_h], axis=1)  # shape (n, 4)
        confidences = obj_scores  # 也可以用 obj_scores * class_scores 作为复合置信度
        class_ids = class_ids

        return detections, confidences, class_ids

    def box_filter(self, frame, result,orig_dim_list=[]):
        orig_img = frame.copy()

        detections = result.boxes.xywhn.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        # detections, confidences, class_ids = self.parse_dets_to_yolov8_format(result, orig_dim_list) # yolov3-spp性能低

        person_detections = []
        person_confidences = []

        for detection, confidence, class_id in zip(detections, confidences, class_ids):
            x1, y1, x2, y2 = yolo2coordinates(detection[0], detection[1], detection[2], detection[3], frame.shape[1], frame.shape[0])
            if class_id == 0 and confidence >= self.conf:
                person_detections.append((x1, y1, x2, y2))
                person_confidences.append(confidence)

        # Draw polygon
        for p1, p2 in zip(self.bound, self.bound[1:] + [self.bound[0]]):
            cv2.line(frame, p1, p2, (68, 70, 254), 2)

        # Create polygon mask directly from points
        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.bound)], 255)

        xyxy_list = []
        conf_list = []

        for (x1, y1, x2, y2), confidence in zip(person_detections, person_confidences):
            bottom_edge_roi = mask[y2 - 2:y2, x1:x2]
            if bottom_edge_roi.size == 0:
                continue
            mask_percentage = np.sum(bottom_edge_roi == 255) / bottom_edge_roi.size
            if mask_percentage > self.limit_percentage:
                xyxy_list.append([x1, y1, x2, y2])
                conf_list.append(confidence)

        xyxy_tensor = torch.tensor(xyxy_list) if xyxy_list else torch.empty((0, 4))
        conf_tensor = torch.tensor(conf_list) if conf_list else torch.empty((0,))

        return orig_img, xyxy_tensor, conf_tensor
