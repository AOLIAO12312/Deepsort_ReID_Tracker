import numpy as np
import cv2

# 读取原始图像
img = cv2.imread('/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/1.png')

# 检查图像是否成功加载
if img is None:
    print("Error: Image not found or unable to load.")
    exit()

# 获取图像尺寸
height, width = img.shape[:2]

# 相机内参矩阵（可以根据你的相机进行调整）
fx, fy = 500, 500  # 焦距
cx, cy = width // 2, height // 2  # 主点（通常为图像中心）

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)  # 相机内参矩阵，确保是 np.float32

# 畸变系数（手动调整这些值）
D = np.array([0.1, -0.2, 0.0, 0.0], dtype=np.float32)  # k1, k2, k3, k4

# 创建窗口
cv2.namedWindow("Fisheye Correction")

# 计算新的相机矩阵
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=0.0)

# 校正图像
undistorted_img = cv2.fisheye.undistortImage(img, K, D, None, new_K)

# 检查校正后的图像是否为空
if undistorted_img is None:
    print("Error: Image undistortion failed.")
    exit()

# 显示校正后的图像
cv2.imshow("Fisheye Correction", undistorted_img)

# 等待按键退出
cv2.waitKey(0)
cv2.destroyAllWindows()
