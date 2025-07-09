import threading
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

from src.camera_fusion import CameraFusion
from src.reid_tracker import ReidTracker
import yaml
from src.bird_eye_view import BirdEyeView
from src.utils import draw_reid_tracking_results
from src.json_writer import JsonWriter

reset_queue = queue.Queue()

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
    # 一次处理15帧
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


def check_contact(tracking_results_group, iou_threshold=0.1, score_threshold=2000):
    """
    计算球员的加权接触评分，若总分超过 `score_threshold`，则判定为关键帧。
    """
    team_A = {"1", "2", "3", "4"}
    team_B = {"5", "6", "7", "8"}

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
                        contact_score += iou * weight  # 关键修改
                        if contact_score >= score_threshold:
                            print(f"camera{camera_idx + 1},id1 = {id1},id2 = {id2},weight = {weight},iou = {iou},score = {contact_score},frame_idx = {frame_idx}")
                            return True  # 立即返回关键帧
    return False  # 低于阈值，不是关键帧

json_writer1 = JsonWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\camera1_short.json")
json_writer2 = JsonWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\camera2_short.json")
json_writer3 = JsonWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\camera3_short.json")
json_writer4 = JsonWriter("E:\\Deepsort_ReID_Tracker\\data\\output\\camera4_short.json")

if __name__ == '__main__':
    camera_name = ["camera1","camera2","camera3","camera4"]

    # 加载目标检测器
    detector_1 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_2 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_3 = YoloDetector(config['yolo']['model_path'], 'cuda:0')
    detector_4 = YoloDetector(config['yolo']['model_path'], 'cuda:0')

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
        'E:\Deepsort_ReID_Tracker\data\input\short_version\Camera1.mp4')
    video_capture_2 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\short_version\Camera2.mp4')
    video_capture_3 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\short_version\Camera3.mp4')
    video_capture_4 = cv2.VideoCapture(
        'E:\Deepsort_ReID_Tracker\data\input\short_version\Camera4.mp4')


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

            list1 = []
            list2 = []
            list3 = []
            list4 = []
            # 遍历一个批次的视频视频数据（15帧）
            for idx, (tracking_results_1,tracking_results_2,tracking_results_3,tracking_results_4) in enumerate(zip(tracking_resultses_1,tracking_resultses_2,tracking_resultses_3,tracking_resultses_4)):
                json_writer1.update(tracking_results_1,frame_idx)
                json_writer2.update(tracking_results_2, frame_idx)
                json_writer3.update(tracking_results_3, frame_idx)
                json_writer4.update(tracking_results_4, frame_idx)


                list1 = []
                list2 = []
                list3 = []
                list4 = []
                # 绘制鸟瞰图
                field1 = bird_view_1.draw_bird_view(tracking_results_1,list1)
                field2 = bird_view_2.draw_bird_view(tracking_results_2,list2)
                field3 = bird_view_3.draw_bird_view(tracking_results_3,list3)
                field4 = bird_view_4.draw_bird_view(tracking_results_4,list4)

                # 更新摄像头数据
                camera_fusion.update_single_camera_position(camera_name[0],list1)
                camera_fusion.update_single_camera_position(camera_name[1],list2)
                camera_fusion.update_single_camera_position(camera_name[2], list3)
                camera_fusion.update_single_camera_position(camera_name[3], list4)

                # 计算摄像头融合数据
                final_position = camera_fusion.fuse_position()

                # 绘制融合数据图像
                mix_field = mix_bird_view.draw_final_pos_view(final_position)


                # 将鸟瞰图放置在帧的左下角用于展示与输出
                monitor_1 = conbine_field_and_frame(field1, frames_1[idx].copy(), tracking_results_1,bound1)
                monitor_2 = conbine_field_and_frame(field2, frames_2[idx].copy(), tracking_results_2,bound2)
                monitor_3 = conbine_field_and_frame(field3, frames_3[idx].copy(), tracking_results_3,bound3)
                monitor_4 = conbine_field_and_frame(field4, frames_4[idx].copy(), tracking_results_4,bound4)

                # 检测碰撞，ret为True则检测到碰撞，False则未检测到，用于关键帧提取
                # group = [tracking_results_1,tracking_results_2,tracking_results_3,tracking_results_4]
                # ret = check_contact(group)

                # if ret is True: # 接触判定代码段
                #     print(f"Keyframe = {frame_idx} extracted")
                #     filename = f"{output_dir}camera1/collision_frame_idx={frame_idx}.jpg"
                #     cv2.imwrite(filename, frames_1[idx])
                #     filename = f"{output_dir}camera2/collision_frame_idx={frame_idx}.jpg"
                #     cv2.imwrite(filename, frames_2[idx])
                #     filename = f"{output_dir}camera3/collision_frame_idx={frame_idx}.jpg"
                #     cv2.imwrite(filename, frames_3[idx])
                #     filename = f"{output_dir}camera4/collision_frame_idx={frame_idx}.jpg"
                #     cv2.imwrite(filename, frames_4[idx])

                # 2 * 2 拼接
                h1 = np.hstack((monitor_1, monitor_2))
                h2 = np.hstack((monitor_3, monitor_4))
                grid = np.vstack((h1, h2))

                # 图像过大，进行放缩
                scale = 0.5
                grid_resized = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                cv2.imshow("Monitor Grid", grid_resized)
                cv2.imshow("mixed-camera",mix_field)
                cv2.waitKey(1)


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