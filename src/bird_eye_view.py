import cv2
import numpy as np
from ultralytics import YOLO
from src.bounding_box_filter import BoundingBoxFilter
from src.utils import get_border

def draw_field():
    # 1. 设定比例尺 & 场地参数
    scale = 50  # 1m = 50px
    width_m, height_m = 13, 10  # 真实场地尺寸（米）

    # 计算场地图像大小（加上边界空白）
    img_width = width_m * scale + 100
    img_height = height_m * scale + 100

    # 创建空白图像（白色背景）
    field = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # 关键坐标计算
    left_x, right_x = 50, 50 + width_m * scale
    top_y, bottom_y = 50, 50 + height_m * scale
    center_x = (left_x + right_x) // 2

    baulk_offset = int(3.75 * scale)
    bonus_offset = int(1 * scale)

    # 绘制卡巴迪场地
    cv2.rectangle(field, (left_x, top_y), (right_x, bottom_y), (0, 0, 0), 2)  # 外边框
    cv2.line(field, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)  # 中线
    cv2.line(field, (center_x - baulk_offset, top_y), (center_x - baulk_offset, bottom_y), (0, 255, 0),
             2)  # Baulk Line
    cv2.line(field, (center_x + baulk_offset, top_y), (center_x + baulk_offset, bottom_y), (0, 255, 0), 2)
    cv2.line(field, (center_x - baulk_offset - bonus_offset, top_y),
             (center_x - baulk_offset - bonus_offset, bottom_y), (255, 0, 0), 1)  # Bonus Line
    cv2.line(field, (center_x + baulk_offset + bonus_offset, top_y),
             (center_x + baulk_offset + bonus_offset, bottom_y), (255, 0, 0), 1)

    return field

class BirdEyeView:
    def __init__(self):
        self.field = draw_field()
        self.bounding_box_filter = None
        self.matrix = None

    def draw_bird_view(self,tracking_results):
        if tracking_results is not None:
            positions = []
            for tracking_result in tracking_results:
                bbox = tracking_result[:4]
                cx = (bbox[0] + bbox[2]) / 2
                # cy = bbox[3] - (bbox[3] - bbox[1]) / 15
                cy = bbox[3]
                positions.append([cx, cy])
            if len(positions) != 0:
                positions = np.array([positions], dtype=np.float32)
                transformed_point = cv2.perspectiveTransform(positions, self.matrix)
                field = self.field.copy()
                for i,point in enumerate(transformed_point[0]):
                    player_x, player_y = point[0], point[1]
                    mapped_x = int(player_x + 50)
                    mapped_y = int(player_y + 50)
                    cv2.circle(field, (mapped_x, mapped_y), 7, (0, 0, 255), -1)  # 画一个半径为7的红色圆
                    # 使用tracking_results[i][4]的数据绘制每个点的ID
                    cv2.putText(field, tracking_results[i][4], (mapped_x + 10, mapped_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                return field
            else:
                return self.field.copy()