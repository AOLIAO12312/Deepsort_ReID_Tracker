import cv2
import numpy as np
from ultralytics import YOLO
from src.bounding_box_filter import BoundingBoxFilter


img = cv2.imread('/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/1.png')
model = YOLO("/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/models/yolo/weights/yolo11n.pt")
result = model(img)[0]

# 显示图像并手动选择四个点
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将选中的点加入列表
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Points", img)

points = []

cv2.imshow("Select Points", img)
cv2.setMouseCallback("Select Points", select_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

# points 现在存储了用户选择的四个点
print("Selected Points:", points)
box_filter = BoundingBoxFilter(points,0.1,0.5)

orig_img,xyxy,conf = box_filter.box_filter(img,result)

positions = []
for bbox in xyxy:
    cx = (bbox[0] + bbox[2])/2
    cy = bbox[3] - (bbox[3]-bbox[1])/15
    positions.append([cx,cy])

positions = np.array([positions],dtype=np.float32)

# 选择目标矩形的四个点
width, height = 650, 500
pts_dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')

# 将选定的四个点转换为numpy数组
pts_src = np.array(points, dtype='float32')

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# 使用透视变换获得新图像
result = cv2.warpPerspective(img, matrix, (width, height))
transformed_point = cv2.perspectiveTransform(positions, matrix)


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
cv2.line(field, (center_x - baulk_offset, top_y), (center_x - baulk_offset, bottom_y), (0, 255, 0), 2)  # Baulk Line
cv2.line(field, (center_x + baulk_offset, top_y), (center_x + baulk_offset, bottom_y), (0, 255, 0), 2)
cv2.line(field, (center_x - baulk_offset - bonus_offset, top_y), (center_x - baulk_offset - bonus_offset, bottom_y), (255, 0, 0), 1)  # Bonus Line
cv2.line(field, (center_x + baulk_offset + bonus_offset, top_y), (center_x + baulk_offset + bonus_offset, bottom_y), (255, 0, 0), 1)

for point in transformed_point[0]:
    # 2. 运动员原始位置 (x, y) 在 650×500 像素的场地中
    player_x, player_y = point[0], point[1]  # 运动员的原始坐标
    # 3. 计算转换后的坐标（+50 边界偏移）
    mapped_x = int(player_x + 50)
    mapped_y = int(player_y + 50)
    # 4. 绘制运动员（红色圆点）
    cv2.circle(field, (mapped_x, mapped_y), 7, (0, 0, 255), -1)  # 画一个半径为7的红色圆

# 显示最终结果
cv2.imshow("Kabaddi Field with Player", field)
cv2.waitKey(0)
cv2.destroyAllWindows()
