import cv2
import numpy as np

# 读取原图
img = cv2.imread('/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/1.png')

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


# 选择目标矩形的四个点
width, height = 700, 1000  # 假设你想要的矩形尺寸
pts_dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype='float32')

# 将选定的四个点转换为numpy数组
pts_src = np.array(points, dtype='float32')

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

# 使用透视变换获得新图像
result = cv2.warpPerspective(img, matrix, (width, height))

# 显示结果
cv2.imshow('Warped Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
