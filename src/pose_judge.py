from collections import deque,defaultdict
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from collections import defaultdict
import os


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


def draw_cluster_distances(cluster, save_path):
    """
    根据 cluster 数据绘制 pose_min_dist 和 coord_dist 曲线图（平滑），并保存到指定路径。

    参数:
        cluster: List[Tuple[frame_idx, (attacker_id, defender_id), pose_dist, coord_dist]]
        save_path: 保存图像的路径（包含 .png 文件名）
    """
    pose_dists = defaultdict(list)
    coord_dists = defaultdict(list)

    # 聚合数据
    for frame_idx, (attacker_id, defender_id), pose_dist, coord_dist in cluster:
        key = (attacker_id, defender_id)
        pose_dists[key].append((frame_idx, pose_dist))
        coord_dists[key].append((frame_idx, coord_dist))

    # 平滑函数
    def smooth_curve(x, y, smooth_points=300):
        if len(x) < 4:
            return x, y
        x = np.array(x)
        y = np.array(y)
        x_new = np.linspace(x.min(), x.max(), smooth_points)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_new)
        return x_new, y_smooth

    # 开始绘图
    plt.figure(figsize=(14, 6))

    # pose_min_dist 子图
    plt.subplot(1, 2, 1)
    for (atk_id, def_id), data in pose_dists.items():
        data.sort()
        frames, values = zip(*data)
        x_smooth, y_smooth = smooth_curve(frames, values)
        plt.plot(x_smooth, y_smooth, label=f'A:{atk_id}, D:{def_id}')
    plt.axhline(50, color='r', linestyle='--', label='Threshold=50')
    plt.title('Pose Min Distance (Smoothed)')
    plt.xlabel('Frame Index')
    plt.ylabel('Pose Min Distance')
    plt.legend()

    # coord_dist 子图
    plt.subplot(1, 2, 2)
    for (atk_id, def_id), data in coord_dists.items():
        data.sort()
        frames, values = zip(*data)
        x_smooth, y_smooth = smooth_curve(frames, values)
        plt.plot(x_smooth, y_smooth, label=f'A:{atk_id}, D:{def_id}')
    plt.axhline(2.0, color='r', linestyle='--', label='Threshold=2.0')
    plt.title('Coordinate Distance (Smoothed)')
    plt.xlabel('Frame Index')
    plt.ylabel('Coordinate Distance')
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

team_A = {"1", "2", "3", "4"}
team_B = {"5", "6", "7", "8"}

class PoseJudge:
    def __init__(self,camera_name):
        self.camera_name = camera_name
        # 存储4个摄像头的关键帧的接触数据
        self.keyframe_contact_recorder = deque(maxlen=60)

        # 定义关键帧簇信息
        self.start_keyframe_idx = 0  # 起始关键帧簇
        self.end_keyframe_idx = 0  # 结束关键帧簇

    def update_keyframe(self,frame_idx,attacker_id,tracking_results,pose_results,bird_view):
        if self.start_keyframe_idx == 0:
            self.start_keyframe_idx = frame_idx  # 初始化起始帧
        self.end_keyframe_idx = frame_idx
        # 将斜视坐标转换为基准坐标（筛选距离相近的人，避免重叠视角的影响）
        positions = []
        for tracking_result in tracking_results:
            bbox = tracking_result[:4]
            cx = (bbox[0] + bbox[2]) / 2
            cy = bbox[3]
            positions.append([cx, cy])
        if len(positions) != 0:
            # 使用对应的转换器进行转换
            transformed_point = bird_view.bbox2coord(positions)  # 实际物理距离需要除以50
        else:
            return

        # 确定进攻者关节点的绝对位置和标准坐标位置
        attacker_absolute_keypoints = None
        for i, tracking_result in enumerate(tracking_results):
            bbox = tracking_result[:4]
            x1, y1, x2, y2 = map(int, bbox)
            id = tracking_result[4]
            if id == attacker_id:
                # 1.2.3 进行关节点判定
                # i 即pose数据中keypoints的索引
                keypoints = pose_results[i]['keypoints']
                # 存放转换后的关键点坐标
                attacker_absolute_keypoints = convert_keypoints_to_absolute(keypoints, x1, y1)
                attacker_x, attacker_y = transformed_point[i][0] / 50, transformed_point[i][1] / 50
                break

        # 2.根据pose_results判定是否接触（关节点位置判定）
        # 通过标准坐标位置数据防止视角重叠导致的误识别
        if attacker_absolute_keypoints is not None:
            for i, tracking_result in enumerate(tracking_results):
                pose_min_dist = None
                coord_dist = None
                idx1, idx2 = None, None
                # 判断是否是对方队伍
                bbox = tracking_result[:4]
                x1, y1, x2, y2 = map(int, bbox)
                id = tracking_result[4]
                current_absolute_keypoints = None
                if id != attacker_id and (attacker_id in team_A and id in team_B) or (
                        attacker_id in team_B and id in team_A):
                    keypoints = pose_results[i]['keypoints']
                    current_absolute_keypoints = convert_keypoints_to_absolute(keypoints, x1, y1)
                    current_x, current_y = transformed_point[i][0] / 50, transformed_point[i][1] / 50

                    # 获取关节点最近距离
                    pose_min_dist, (idx1, idx2) = find_closest_keypoint_index(attacker_absolute_keypoints,
                                                                              current_absolute_keypoints)
                    # 标准坐标距离
                    coord_dist = np.linalg.norm([current_x - attacker_x, current_y - attacker_y])
                    # pose_min_dist为像素距离（像素）  coord_dist为物理距离（米）
                    # print(f'attacker_id:{attacker_id},defender_id:{id} pose_min_dist:{pose_min_dist},coord_dist:{coord_dist}')

                    # 写入数据，格式：frame_idx,(attacker_id,defender_id),pose_min_dist,coord_dist 作为关键帧分析数据
                    # 循环外对该数据进行分析，一个关键帧组（连续提取的多个关键帧），找到pose_min_dist,coord_dist在阈值下且最小值的frame_idx进行输出，其他关键帧舍去，判定为接触
                    # 会延迟输出
                    # 测试第二个摄像头
                    self.keyframe_contact_recorder.append((frame_idx, (attacker_id, id), pose_min_dist, coord_dist))

    def check_keyframe_cluster(self,frame_idx):
        # 对关键帧簇进行处理
        if self.end_keyframe_idx - self.start_keyframe_idx >= 8 and frame_idx - self.end_keyframe_idx > 10:
            # 2.当前的frame_idx-end_keyframe_idx > 15（即当前簇已记录完） 时提醒进行数据处理
            cluster = list(self.keyframe_contact_recorder)
            result = []
            pair_to_contacts = defaultdict(list)
            save_path = f'data/output/keyframe/distance_analyze/{self.camera_name}/cluster_{self.start_keyframe_idx}_{self.end_keyframe_idx}.png'
            draw_cluster_distances(cluster, save_path)
            # 1. 收集符合阈值的接触
            for frame_idx, (attacker_id, defender_id), pose_dist, coord_dist in cluster:
                # 在此处插入图像绘制程序并进行保存，并标注好筛选阈值
                if pose_dist < 50 and coord_dist < 2:  # 阈值可以调整
                    pair_to_contacts[(attacker_id, defender_id)].append(
                        (frame_idx, (attacker_id, defender_id), pose_dist, coord_dist)
                    )



            # 2. 对每个对手组合选出最小的接触（pose优先，其次coord）
            for pair, contacts in pair_to_contacts.items():
                min_contact = min(contacts, key=lambda x: (x[2], x[3]))  # x[2] 是 pose_dist, x[3] 是 coord_dist
                result.append({
                    'cluster_range': (cluster[0][0], cluster[-1][0]),
                    'best_frame': min_contact[0],
                    'pair': min_contact[1],
                    'pose_dist': min_contact[2],
                    'coord_dist': min_contact[3],
                    # 'all_contacts': contacts  # 如有需要可保留所有contact
                })
            print(self.camera_name)
            print(result)
            # 处理结束，清理关键帧数据
            self.keyframe_contact_recorder.clear()
            # 重置关键帧簇
            self.start_keyframe_idx = 0
            self.end_keyframe_idx = 0