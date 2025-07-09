import time

import torch

import pose_detector

import os
import cv2


def load_png_images_from_folder(folder_path):
    """
    读取指定文件夹下所有 PNG 图片，返回图像列表。

    :param folder_path: str，图片文件夹路径
    :return: List[np.ndarray]，图像数组列表
    """
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)  # 读取为BGR格式图像
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Failed to read image {img_path}")
    return images




if __name__ == '__main__':
    pose_detector = pose_detector.PoseDetector()

    folder = r"E:\Deepsort_ReID_Tracker\data\input\Athlete\5"
    frames = load_png_images_from_folder(folder)

    print(f"共读取到 {len(frames)} 张图片")

    start = time.time()
    results = pose_detector.get_result(frames)  # 返回 keypoints 数据
    end = time.time()
    print(f"函数 {pose_detector.get_result.__name__} 执行耗时: {end - start:.6f} 秒")

    for i, frame in enumerate(frames):
        keypoints = results[i]['keypoints']  # shape: (136, 2)
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()

        for x, y in keypoints:
            # 只画合法坐标点（>0）
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

        cv2.imshow(f"Pose {i+1}", frame)
        key = cv2.waitKey(0)  # 按任意键继续
        if key == 27:  # ESC退出
            break
        cv2.destroyAllWindows()






