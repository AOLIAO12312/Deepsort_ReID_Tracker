import cv2
import numpy as np

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

# 2. 运动员原始位置 (x, y) 在 650×500 像素的场地中
player_x, player_y = 325, 250  # 运动员的原始坐标

# 3. 计算转换后的坐标（+50 边界偏移）
mapped_x = player_x + 50
mapped_y = player_y + 50

# 4. 绘制运动员（红色圆点）
cv2.circle(field, (mapped_x, mapped_y), 10, (0, 0, 255), -1)  # 画一个半径为10的红色圆

# 显示最终结果
cv2.imshow("Kabaddi Field with Player", field)
cv2.waitKey(0)
cv2.destroyAllWindows()
