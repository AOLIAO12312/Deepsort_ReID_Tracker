import threading
from collections import deque, defaultdict

import numpy as np
import torch
from tqdm import tqdm
from types import SimpleNamespace
from alphapose.utils.config import update_config # 读取配置文件
from alphapose.utils.detector import DetectionLoader
from detector.apis import get_detector


from models.yolo.yolo_detector import YoloDetector
import cv2
import queue

from src.pose_judge import PoseJudge
from src.state_analyzer import save_person_state_to_csv

from src.camera_fusion import CameraFusion
from src.reid_tracker import ReidTracker
import yaml
from src.bird_eye_view import BirdEyeView
from src.state_analyzer import StateAnalyzer
from src.utils import draw_reid_tracking_results
from src.json_writer import JsonWriter
from src.pose_detector import PoseDetector

reset_queue = queue.Queue()

team_A = {"1", "2", "3", "4"}
team_B = {"5", "6", "7", "8"}

def load_config(config_path:str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def wait_for_input():
    while True:
        user_input = input("Please enter the person needs to be reset:")
        reset_queue.put(user_input)

config = load_config("E:\Deepsort_ReID_Tracker\configs\config.yaml")

camera1_results_queue = queue.Queue()
camera2_results_queue = queue.Queue()
camera3_results_queue = queue.Queue()
camera4_results_queue = queue.Queue()

camera1_reset_queue = queue.Queue()
camera2_reset_queue = queue.Queue()
camera3_reset_queue = queue.Queue()
camera4_reset_queue = queue.Queue()

camera1_fix_queue = queue.Queue()
camera2_fix_queue = queue.Queue()
camera3_fix_queue = queue.Queue()
camera4_fix_queue = queue.Queue()

def read_frame_and_track(video_capture,reid_tracker,camera_results_queue):
    ret = True
    frames = []
    # 一次处理10帧
    for i in range(15):
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    if not ret:
        camera_results_queue.put((None, None))
    tracking_resultses = reid_tracker.multi_frame_update(frames)
    camera_results_queue.put((frames,tracking_resultses))

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
    field_resized = cv2.resize(field, (0, 0), fx=scale_factor, fy=scale_factor)
    fh, fw, _ = frame.shape
    th, tw, _ = field_resized.shape
    x_offset = 10
    y_offset = fh - th - 10
    roi = frame[y_offset:y_offset + th, x_offset:x_offset + tw]
    alpha = 0.8
    blended = cv2.addWeighted(field_resized, alpha, roi, 1 - alpha, 0)
    frame[y_offset:y_offset + th, x_offset:x_offset + tw] = blended
    return frame


def compute_iou(box1, box2):
    """
    Intersection over Union
    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1, inter_y1 = max(x1, x3), max(y1, y3)
    inter_x2, inter_y2 = min(x2, x4), min(y2, y4)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0, area1, area2  # 额外返回框面积


def check_contact(tracking_results_group, iou_threshold=0.1, score_threshold=3500):
    """
    计算球员的加权接触评分，若总分超过 `score_threshold`，则判定为关键帧。
    """

    contact_score = 0  # 记录总接触评分
    for camera_idx,tracking_results in enumerate(tracking_results_group):
        all_detections = list(tracking_results)
        for i in range(len(all_detections)):
            box1, id1 = all_detections[i][:4], str(all_detections[i][4])
            for j in range(i + 1, len(all_detections)):
                box2, id2 = all_detections[j][:4], str(all_detections[j][4])
                if (id1 in team_A and id2 in team_B) or (id2 in team_A and id1 in team_B):
                    iou, area1, area2 = compute_iou(box1, box2)
                    if iou > iou_threshold:
                        weight = min(area1, area2)  # 取较小目标框
                        contact_score += iou * weight  # 计算接触评分
                        if contact_score >= score_threshold:
                            # print(f"camera{camera_idx + 1},id1 = {id1},id2 = {id2},weight = {weight},iou = {iou},score = {contact_score},frame_idx = {frame_idx}")
                            return True  # 立即返回关键帧
    return False  # 低于阈值，不是关键帧


def frame_pose_detection(frame,tracking_results,pose_detector):
    """对输入的帧检测每个目标的姿态"""
    if len(tracking_results) == 0:
        return frame, None
    # 1.截取需要姿态检测的裁切图片
    person_pic = []
    pose_results = []
    for tracking_result in tracking_results:
        bbox = tracking_result[:4]
        id = tracking_result[4]
        x1, y1, x2, y2 = map(int, bbox)  # 确保是整数索引
        cropped_image = frame[y1:y2, x1:x2]  # 注意：先y后x
        person_pic.append((cropped_image, id))
    # 2.推理姿态
    pose_results = pose_detector.get_result(img for img, _ in person_pic)

    # # 3.1 在每个裁切图上绘制关节点(测试用)
    # for i,(frame,id) in enumerate(person_pic_1):
    #     keypoints = pose_results_1[i]['keypoints']  # shape: (26, 2)
    #     if isinstance(keypoints, torch.Tensor):
    #         keypoints = keypoints.detach().cpu().numpy()
    #
    #     for x, y in keypoints:
    #         # 只画合法坐标点（>0）
    #         if x > 0 and y > 0:
    #             cv2.circle(frame, (int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)
    #
    #     cv2.imshow(f"Pose {id}", frame)
    #     key = cv2.waitKey(0)  # 按任意键继续
    #     if key == 27:  # ESC退出
    #         break
    #     cv2.destroyAllWindows()

    # 3.2 在帧上绘制汇总的关节点（运行时输出关节点图像）
    for i, tracking_result in enumerate(tracking_results):
        bbox = tracking_result[:4]
        id = tracking_result[4]
        x1, y1, x2, y2 = map(int, bbox)
        keypoints = pose_results[i]['keypoints']  # shape: (26, 2)
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()
        draw_reid_tracking_results(tracking_results, frame)
        for x, y in keypoints:
            # 只画合法坐标点（>0）
            if int(x) + x1 > 0 and int(y) + y1 > 0:
                cv2.circle(frame, (int(x) + x1, int(y) + y1), radius=2, color=(0, 255, 0), thickness=-1)
    # cv2.imshow(f"Keyframe{idx},Camera_1", frame)
    # key = cv2.waitKey(0)  # 按任意键继续
    # if key == 27:  # ESC退出
    #     return
    # cv2.destroyAllWindows()
    return frame, pose_results




import numpy as np

def assign_unknown_ids_by_mapping(tracking_results, final_positions, bird_view,
                                   offset=(50, 50), distance_threshold=10000):
    """
    给分视角 tracking_results 中的 Unknown 目标分配 ID，避免重复使用其他摄像头或本摄像头已有的 ID。

    参数:
    - tracking_results: List[List]，每个元素为 [x1, y1, x2, y2, id]
    - final_positions: List[Tuple[float, float, str]]，融合后的全局坐标 + id
    - bird_view: 具有 .bbox2coord() 方法的对象，用于视角坐标 → 全局映射
    - offset: Tuple[int, int]，可用于微调映射坐标，如 (50, 50)
    - distance_threshold: float，最大允许匹配距离平方，避免误配
    - used_ids: set，其他摄像头已分配的 ID，避免冲突

    返回:
    - updated_tracking_results: List[List]，填充后的 tracking_results
    - newly_used_ids: set，新分配的 ID 集合（用于更新 used_ids）
    """
    newly_used_ids = set()
    offset_x, offset_y = offset

    # 收集当前摄像头中已存在的 ID（非 Unknown）
    existing_ids_in_current_frame = set(
        str(result[4]) for result in tracking_results if not str(result[4]).startswith("U")
    )

    for tracking_result in tracking_results:
        bbox = tracking_result[:4]
        original_id = tracking_result[4]

        if str(original_id).startswith("U"):
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]  # bottom center
            transformed_point = bird_view.bbox2coord([[cx, cy]])
            mapped_x = int(transformed_point[0][0] + offset_x)
            mapped_y = int(transformed_point[0][1] + offset_y)

            closest_id = None
            closest_distance_sq = float('inf')

            for final_x, final_y, final_id in final_positions:
                if (final_id in newly_used_ids or
                    final_id in existing_ids_in_current_frame):
                    continue  # 已被使用或本摄像头已存在

                dist_sq = (final_x - mapped_x)**2 + (final_y - mapped_y)**2
                if dist_sq < closest_distance_sq:
                    closest_distance_sq = dist_sq
                    closest_id = final_id

            if closest_id is not None and closest_distance_sq < distance_threshold:
                tracking_result[4] = closest_id
                newly_used_ids.add(closest_id)

    return tracking_results

def convert_keypoints_to_absolute(keypoints, x1, y1):
    """
    将相对裁剪图中的 keypoints 坐标转换为整张图中的绝对坐标。

    参数:
        keypoints: numpy.ndarray 或 torch.Tensor，形状为 (N, 2) 或 (N, 3)，代表 N 个关键点的 (x, y) 或 (x, y, score)
        x1, y1: 裁剪区域左上角在整图中的坐标

    返回:
        absolute_keypoints: list of (x, y)，绝对坐标形式的关键点（只保留坐标都大于 0 的）
    """
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.detach().cpu().numpy()

    absolute_keypoints = []

    for kp in keypoints:
        # 兼容 (x, y) 或 (x, y, score)
        if len(kp) >= 2:
            x, y = kp[:2]
            abs_x = int(x) + x1
            abs_y = int(y) + y1
            if abs_x > 0 and abs_y > 0:
                absolute_keypoints.append((abs_x, abs_y))

    return absolute_keypoints


import numpy as np

import numpy as np

def find_closest_keypoint_index(kpts1, kpts2):
    """
    查找两组关键点中距离最近的一对，并返回它们的索引和距离。

    参数:
        kpts1: list of (x, y)，第一组关键点
        kpts2: list of (x, y)，第二组关键点

    返回:
        min_distance: float，最小距离
        idx_pair: tuple (idx1, idx2)，分别是kpts1和kpts2中最近关键点的索引
    """
    min_distance = float('inf')
    idx_pair = (-1, -1)

    for i, p1 in enumerate(kpts1):
        for j, p2 in enumerate(kpts2):
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < min_distance:
                min_distance = dist
                idx_pair = (i, j)

    return min_distance, idx_pair



# json_writer1 = JsonWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\camera1_short.json")
# json_writer2 = JsonWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\camera2_short.json")
# json_writer3 = JsonWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\camera3_short.json")
# json_writer4 = JsonWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\camera4_short.json")

if __name__ == '__main__':
    camera_name = ["camera1","camera2","camera3","camera4"]
    pose_detector = PoseDetector()
    state_analyzer = StateAnalyzer()

    # 加载目标检测器
    detector_1 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_2 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_3 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_4 = YoloDetector(config['yolo']['model_path'], 'cuda:0')

    pose_judge_1 = PoseJudge(camera_name[0])
    pose_judge_2 = PoseJudge(camera_name[1])
    pose_judge_3 = PoseJudge(camera_name[2])
    pose_judge_4 = PoseJudge(camera_name[3])

    # 加载鸟瞰图生成器
    bird_view_1 = BirdEyeView()
    bird_view_2 = BirdEyeView()
    bird_view_3 = BirdEyeView()
    bird_view_4 = BirdEyeView()
    mix_bird_view=BirdEyeView()

    # 加载人物跟踪识别器
    reid_tracker_1 = ReidTracker(detector_1,config['deepsort']['deepsort_cfg_path'],
                                                'E:\Deepsort_ReID_Tracker\data\input\Camera1_base_data',config, camera1_reset_queue,camera1_fix_queue,camera_name[0],'cuda:0')
    reid_tracker_2 = ReidTracker(detector_2, config['deepsort']['deepsort_cfg_path'],
                                 'E:\Deepsort_ReID_Tracker\data\input\Camera2_base_data', config, camera2_reset_queue,camera2_fix_queue ,camera_name[1],'cuda:0')
    reid_tracker_3 = ReidTracker(detector_3, config['deepsort']['deepsort_cfg_path'],
                                 'E:\Deepsort_ReID_Tracker\data\input\Camera3_base_data', config, camera3_reset_queue,camera3_fix_queue,camera_name[2] ,'cuda:0')
    reid_tracker_4 = ReidTracker(detector_4, config['deepsort']['deepsort_cfg_path'],
                                 'E:\Deepsort_ReID_Tracker\data\input\Camera4_base_data', config, camera4_reset_queue,camera4_fix_queue, camera_name[3],'cuda:0')

    # 初始化视频输入流
    video_capture_1 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\long_video\Camera1.mp4')
    video_capture_2 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\long_video\Camera2.mp4')
    video_capture_3 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\long_video\Camera3.mp4')
    video_capture_4 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\long_video\Camera4.mp4')


    camera_fusion = CameraFusion()

    total_frames = int(video_capture_1.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = 'E:/Deepsort_ReID_Tracker/data/output/ablation_output/tracked_video_MIXED_short.mp4'
    output_video_path2 = 'E:/Deepsort_ReID_Tracker/data/output/ablation_output/tracked_video_GRID_short.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (750, 600))
    video_writer2 = cv2.VideoWriter(output_video_path2, fourcc, 30, (frame_width, frame_height))

    video_writer_camera1 = cv2.VideoWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\ablation_output\\camera1_short.mp4",
                                           fourcc,30,(frame_width, frame_height))
    video_writer_camera2 = cv2.VideoWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\ablation_output\\camera2_short.mp4",
                                           fourcc, 30, (frame_width, frame_height))
    video_writer_camera3 = cv2.VideoWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\ablation_output\\camera3_short.mp4",
                                           fourcc, 30, (frame_width, frame_height))
    video_writer_camera4 = cv2.VideoWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\ablation_output\\camera4_short.mp4",
                                           fourcc, 30, (frame_width, frame_height))


    # input_thread = threading.Thread(target=wait_for_input, daemon=True)
    # input_thread.start()

    # 帧数记录
    frame_idx = 0
    output_dir = f"E:/Deepsort_ReID_Tracker/data/output/keyframe/"

    # bound初始化
    bound1 = []
    bound2 = []
    bound3 = []
    bound4 = []

    process_finish = threading.Event()

    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while True:
            # 创建4个线程用于读取四个机位的视频数据，并执行人物跟踪与识别
            track_thread_1 = threading.Thread(target=read_frame_and_track,
                                              args=(video_capture_1,reid_tracker_1,camera1_results_queue))
            track_thread_2 = threading.Thread(target=read_frame_and_track,
                                              args=(video_capture_2, reid_tracker_2, camera2_results_queue))
            track_thread_3 = threading.Thread(target=read_frame_and_track,
                                              args=(video_capture_3, reid_tracker_3, camera3_results_queue))
            track_thread_4 = threading.Thread(target=read_frame_and_track,
                                              args=(video_capture_4, reid_tracker_4, camera4_results_queue))

            # 启动线程
            track_thread_1.start()
            track_thread_2.start()
            track_thread_3.start()
            track_thread_4.start()

            # 等待4个线程处理完毕
            track_thread_1.join()
            track_thread_2.join()
            track_thread_3.join()
            track_thread_4.join()

            # 从4个摄像头的结果数据队列中读取视频数据
            frames_1,tracking_resultses_1 = camera1_results_queue.get()
            frames_2,tracking_resultses_2 = camera2_results_queue.get()
            frames_3,tracking_resultses_3 = camera3_results_queue.get()
            frames_4,tracking_resultses_4 = camera4_results_queue.get()


            # 若返回None则表示视频读取结束
            if frames_1 is None or frames_2 is None or frames_3 is None or frames_4 is None:
                break

            # 第0帧时获取filter的bound数据,初始化从tracker中获取鸟瞰图的生成器的视角转换矩阵
            if frame_idx == 0:
                bound1 = reid_tracker_1.bounding_box_filter.bound
                bound2 = reid_tracker_2.bounding_box_filter.bound
                bound3 = reid_tracker_3.bounding_box_filter.bound
                bound4 = reid_tracker_4.bounding_box_filter.bound
                bird_view_1.matrix = reid_tracker_1.matrix
                bird_view_2.matrix = reid_tracker_2.matrix
                bird_view_3.matrix = reid_tracker_3.matrix
                bird_view_4.matrix = reid_tracker_4.matrix


            # 遍历一个批次的视频视频数据（15帧）
            for idx, (tracking_results_1,tracking_results_2,tracking_results_3,tracking_results_4) in enumerate(zip(tracking_resultses_1,tracking_resultses_2,tracking_resultses_3,tracking_resultses_4)):
                # json_writer1.update(tracking_results_1,frame_idx)
                # json_writer2.update(tracking_results_2, frame_idx)
                # json_writer3.update(tracking_results_3, frame_idx)
                # json_writer4.update(tracking_results_4, frame_idx)

                list1 = []
                list2 = []
                list3 = []
                list4 = []

                # 绘制鸟瞰图
                field1 = bird_view_1.draw_bird_view(tracking_results_1, list1)
                field2 = bird_view_2.draw_bird_view(tracking_results_2, list2)
                field3 = bird_view_3.draw_bird_view(tracking_results_3, list3)
                field4 = bird_view_4.draw_bird_view(tracking_results_4, list4)

                # 更新摄像头数据
                camera_fusion.update_single_camera_position(camera_name[0], list1)
                camera_fusion.update_single_camera_position(camera_name[1], list2)
                camera_fusion.update_single_camera_position(camera_name[2], list3)
                camera_fusion.update_single_camera_position(camera_name[3], list4)

                # 计算摄像头融合数据
                final_position = camera_fusion.fuse_position()

                # 更新场景数据进行比赛分析
                state_analyzer.update(final_position)

                persons_state = state_analyzer.get_persons_state()

                save_person_state_to_csv(persons_state, frame_idx, "data/output/keyframe/person_states.csv",
                                         first_write=state_analyzer.first_write)
                state_analyzer.first_write = False

                # for person in persons_state:
                #     print(f"frame_idx {frame_idx}")
                #     print(f"运动员 ID: {person.id}")
                #     print(f"  位置: {person.position}")
                #     print(f"  速度: {person.velocity:.2f} m/s")
                #     print(f"  加速度: {person.acceleration:.2f} m/s²")
                #     print(f"  方向: {person.direction:.1f}°")
                #     print(f"  状态: {person.state.name}")  # GameState 是枚举，取 name 字符串更清晰
                #     print("-" * 40)

                # 尝试使用融合坐标数据填充/修正原始摄像头数据的缺失值和误差值,尽量保证每个分摄像头的数据的完整性（减少Unknown目标）
                tracking_results_1 = assign_unknown_ids_by_mapping(
                    tracking_results_1, final_position, bird_view_1
                )
                tracking_results_2 = assign_unknown_ids_by_mapping(
                    tracking_results_2, final_position, bird_view_2
                )
                tracking_results_3 = assign_unknown_ids_by_mapping(
                    tracking_results_3, final_position, bird_view_3
                )
                tracking_results_4 = assign_unknown_ids_by_mapping(
                    tracking_results_4, final_position, bird_view_4
                )

                # 绘制融合数据图像
                mix_field = mix_bird_view.draw_final_pos_view(final_position)

                # 将鸟瞰图放置在帧的左下角用于展示与输出
                monitor_1 = conbine_field_and_frame(field1, frames_1[idx].copy(), tracking_results_1,bound1)
                monitor_2 = conbine_field_and_frame(field2, frames_2[idx].copy(), tracking_results_2,bound2)
                monitor_3 = conbine_field_and_frame(field3, frames_3[idx].copy(), tracking_results_3,bound3)
                monitor_4 = conbine_field_and_frame(field4, frames_4[idx].copy(), tracking_results_4,bound4)

                # 检测碰撞，ret为True则检测到碰撞，False则未检测到，用于关键帧提取
                group = [tracking_results_1,tracking_results_2,tracking_results_3,tracking_results_4]
                ret = check_contact(group)

                if ret is True: # 接触判定代码段
                    # 执行姿态检测
                    # 输出的是原始检测数据，非拼接整合数据，需自行处理
                    pose_frame_1, pose_results_1 = frame_pose_detection(frames_1[idx].copy(), tracking_results_1,
                                                                        pose_detector)
                    pose_frame_2, pose_results_2 = frame_pose_detection(frames_2[idx].copy(), tracking_results_2,
                                                                        pose_detector)
                    pose_frame_3, pose_results_3 = frame_pose_detection(frames_3[idx].copy(), tracking_results_3,
                                                                        pose_detector)
                    pose_frame_4, pose_results_4 = frame_pose_detection(frames_4[idx].copy(), tracking_results_4,
                                                                        pose_detector)

                    # TODO:已知final_position的运动员位置和四个摄像头的姿态数据，依据此进行接触判定（四机位俯视+斜视）
                    # 已有 pose_results（姿态+身份数据） ， tracking_results（目标框+身份数据），orig_img原始图像数据
                    # 1.筛选距离相近的异队运动员（依据final_position判定）（运动员靠近是基础条件）
                    # 1.1 对于每个摄像头进行筛选，而不是融合数据（缺失可能性大）
                    # 1.2 筛选出可能的碰撞的运动员
                    # 1.2.1 根据person_state数据找到进攻队员
                    attacker_id = state_analyzer.get_current_attacker()


                    # 集成，进行姿态判定
                    pose_judge_1.update_keyframe(frame_idx, attacker_id, tracking_results_1, pose_results_1,
                                                 bird_view_1)
                    pose_judge_2.update_keyframe(frame_idx, attacker_id, tracking_results_2, pose_results_2,
                                                 bird_view_2)
                    pose_judge_3.update_keyframe(frame_idx, attacker_id, tracking_results_3, pose_results_3,
                                                 bird_view_3)
                    pose_judge_4.update_keyframe(frame_idx, attacker_id, tracking_results_4, pose_results_4,
                                                 bird_view_4)

                    # 1.2.2 在每个摄像头中筛选进攻队员（没有则舍弃，有则保留）

                    # 将斜视坐标转换为基准坐标（筛选距离相近的人，避免重叠视角的影响）
                    # 3.需要根据语义分割模型进行细化判定（视角重叠仍难以解决）（暂时搁置）


                    # 保存姿态检测结果（图片）
                    # print(f"Keyframe = {frame_idx} extracted")
                    filename = f"{output_dir}camera1/collision_frame_idx={frame_idx}.jpg"
                    cv2.imwrite(filename, pose_frame_1)
                    filename = f"{output_dir}camera2/collision_frame_idx={frame_idx}.jpg"
                    cv2.imwrite(filename, pose_frame_2)
                    filename = f"{output_dir}camera3/collision_frame_idx={frame_idx}.jpg"
                    cv2.imwrite(filename, pose_frame_3)
                    filename = f"{output_dir}camera4/collision_frame_idx={frame_idx}.jpg"
                    cv2.imwrite(filename, pose_frame_4)

                # 后处理关键帧数据
                # if end_keyframe_idx != 0 and start_keyframe_idx != 0 and end_keyframe_idx - start_keyframe_idx < 8:
                #     # 两者间隔不超过8直接舍弃 （关键帧簇不可太短）
                #     start_keyframe_idx = 0
                #     end_keyframe_idx = 0

                pose_judge_1.check_keyframe_cluster(frame_idx)
                pose_judge_2.check_keyframe_cluster(frame_idx)
                pose_judge_3.check_keyframe_cluster(frame_idx)
                pose_judge_4.check_keyframe_cluster(frame_idx)

                # 2 * 2 拼接
                h1 = np.hstack((monitor_1, monitor_2))
                h2 = np.hstack((monitor_3, monitor_4))
                grid = np.vstack((h1, h2))

                # 图像过大，进行放缩
                scale = 0.5
                grid_resized = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                # 显示实时运行结果
                # cv2.imshow("Monitor Grid", grid_resized)
                # cv2.imshow("mixed-camera",mix_field)
                # cv2.waitKey(1)


                # 写入视频数据
                video_writer2.write(grid_resized)
                video_writer.write(mix_field)
                video_writer_camera1.write(monitor_1)
                video_writer_camera2.write(monitor_2)
                video_writer_camera3.write(monitor_3)
                video_writer_camera4.write(monitor_4)
                frame_idx += 1

            camera_fusion.find_difference_and_feedback(list1,
                                                       camera1_reset_queue,camera1_fix_queue,'camera1')
            camera_fusion.find_difference_and_feedback(list2,
                                                       camera2_reset_queue,camera2_fix_queue,'camera2')
            camera_fusion.find_difference_and_feedback(list3,
                                                       camera3_reset_queue,camera3_fix_queue,'camera3')
            camera_fusion.find_difference_and_feedback(list4,
                                                       camera4_reset_queue,camera4_fix_queue,'camera4')

            pbar.update(15)
    process_finish.set()
    video_capture_1.release()
    video_capture_2.release()
    video_capture_3.release()
    video_capture_4.release()
    video_writer.release()
    video_writer2.release()
    video_writer_camera1.release()
    video_writer_camera2.release()
    video_writer_camera3.release()
    video_writer_camera4.release()
    cv2.destroyAllWindows()
    print(f"Processing complete, video has been saved as: {output_video_path} and {output_video_path2}")