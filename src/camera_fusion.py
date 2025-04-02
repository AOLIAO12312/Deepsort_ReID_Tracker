import numpy as np
import math
from sklearn.linear_model import LinearRegression
from collections import defaultdict, deque


def find_closest_point(point_list, target_point):
    """
    该函数用于找出列表中距离目标点最近的点
    :param point_list: 包含多个点的列表，每个点以元组 (x, y) 形式表示
    :param target_point: 目标点，以元组 (x, y) 形式表示
    :return: 距离目标点最近的点
    """
    min_distance = float('inf')  # 初始化最小距离为正无穷大
    closest_point = None  # 初始化最近的点为 None

    for point in point_list:
        # 计算当前点与目标点之间的欧几里得距离
        distance = math.sqrt((point[0] - target_point[0]) ** 2 + (point[1] - target_point[1]) ** 2)
        if distance < min_distance:
            # 如果当前距离小于最小距离，则更新最小距离和最近的点
            min_distance = distance
            closest_point = point

    return closest_point

def calculate_distance(point1, point2):
    """
    计算平面上两点之间的欧几里得距离
    :param point1: 第一个点的坐标，格式为 (x1, y1)
    :param point2: 第二个点的坐标，格式为 (x2, y2)
    :return: 两点之间的距离
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_outliers_std(data, k=3):
    mean_value = np.mean(data)
    std_value = np.std(data)
    lower_bound = mean_value - k * std_value
    upper_bound = mean_value + k * std_value
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    # print("outliers:",outliers)
    return outliers


# def vote_camera(list1,list2,list3,list4,x_coords,y_coords):
#     pos1=[]
#     pos2=[]
#     pos3=[]
#     pos4=[]
#     pos5=[]
#     pos6=[]
#     pos7=[]
#     pos8=[]
#     pos_list=[pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]
#
#
#     for i in range(len(list1)):
#         for j in range(8):
#             value=list1[i][2]
#             if len(value)==1:
#                 if 1<=int(value)<=8:
#                     temp=(list1[i][0],list1[i][1])
#                     if temp not in pos_list[int(value)-1]:
#                         pos_list[int(value)-1].append(temp)
#     for i in range(len(list2)):
#         for j in range(8):
#             value=list2[i][2]
#             if len(value)==1:
#                 if 1<=int(value)<=8:
#                     temp = (list2[i][0], list2[i][1])
#                     if temp not in pos_list[int(value) - 1]:
#                         pos_list[int(value) - 1].append(temp)
#     for i in range(len(list3)):
#         for j in range(8):
#             value=list3[i][2]
#             if len(value)==1:
#                 if 1<=int(value)<=8:
#                     temp = (list3[i][0], list3[i][1])
#                     if temp not in pos_list[int(value) - 1]:
#                         pos_list[int(value) - 1].append(temp)
#     for i in range(len(list4)):
#         for j in range(8):
#             value=list4[i][2]
#             if len(value) == 1:
#                 if 1<=int(value)<=8:
#                     temp = (list4[i][0], list4[i][1])
#                     if temp not in pos_list[int(value) - 1]:
#                         pos_list[int(value) - 1].append(temp)
#     # for idx, sub_list in enumerate(pos_list, start=1):
#     #     print(f"pos{idx}: {sub_list}")
#
#     #对每个球员的位置坐标进行投票
#     final_pos = []
#     window_size=5
#     predicted_x=None
#     predicted_y=None
#     for idx in range(8):
#         x_all=[]
#         y_all=[]
#         xy_all=[]
#         closet_point=[]
#         for id in pos_list[idx]:
#             x_all.append(id[0])
#             y_all.append(id[1])
#             xy_all.append((id[0],id[1]))
#         # 做坐标线性预测
#         # if len(x_coords[idx])==5 and len(y_coords[idx])==5:
#         #     x_coordss = np.array(x_coords[idx])
#         #     y_coordss = np.array(y_coords[idx])
#         #     last_x_coords = x_coordss[-window_size:].reshape(-1, 1)
#         #     last_y_coords = y_coordss[-window_size:].reshape(-1, 1)
#         #     last_frames = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
#         #     x_model = LinearRegression()
#         #     x_model.fit(last_frames, last_x_coords)
#         #
#         #     y_model = LinearRegression()
#         #     y_model.fit(last_frames, last_y_coords)
#         #
#         #     next_frame = np.array([6]).reshape(-1, 1)
#         #     predicted_x = x_model.predict(next_frame)
#         #     predicted_y = y_model.predict(next_frame)
#             # print("predicted x,y:",predicted_x[0][0],predicted_y[0][0])
#         # std_x=np.std(x_all)
#         if len(x_coords[idx])==5:
#             x_frames = np.array(x_coords[idx])
#             y_frames = np.array(y_coords[idx])
#             dt = 1/30  # 时间间隔
#
#             # 构建时间序列
#             t = np.array([i * dt for i in range(len(x_frames))]).reshape(-1, 1)
#
#             # 对 x 坐标进行线性回归
#             reg_x = LinearRegression().fit(t, x_frames.reshape(-1, 1))
#             vx = reg_x.coef_[0][0]
#
#             # 对 y 坐标进行线性回归
#             reg_y = LinearRegression().fit(t, y_frames.reshape(-1, 1))
#             vy = reg_y.coef_[0][0]
#
#             # 当前帧为最后一帧
#             xn = x_frames[-1]
#             yn = y_frames[-1]
#             # print(vx,vy)
#             # 预测下一帧坐标
#             predicted_x = xn + vx * dt
#             predicted_y = yn + vy * dt
#
#             # print(f"预测下一帧的 坐标: {predicted_x},{predicted_y}")
#
#             # dt = 1/30  # 时间间隔
#             # x2=x_coords[idx][1]
#             # x1=x_coords[idx][0]
#             # y1=y_coords[idx][1]
#             # y2=y_coords[idx][0]
#
#             # 计算速度
#             # vx = (x2 - x1) / dt
#             # vy = (y2 - y1) / dt
#             #
#             # # 预测下一帧坐标
#             # predicted_x = x2 + vx * dt
#             # predicted_y = y2 + vy * dt
#             #
#             # print(f"预测下一帧的 x 坐标: {predicted_x}")
#             # print(f"预测下一帧的 y 坐标: {predicted_y}")
#         if x_all:
#
#             ave_x=np.average(x_all)
#             ave_y=np.average(y_all)
#             # final_pos.append([ave_x,ave_y,idx+1])
#             # # 保存x的序列，滑动窗口
#             # x_coords[idx].append(int(ave_x))
#             # if len(x_coords[idx]) > window_size:
#             #     x_coords[idx].pop(0)
#             # # 保存y的序列，滑动窗口
#             # y_coords[idx].append(int(ave_y))
#             # if len(y_coords[idx])>window_size:
#             #     y_coords[idx].pop(0)
#
#             # print("xy_all",xy_all)
#             if predicted_x:
#                 # print("最近的点:",find_closest_point(xy_all,(predicted_x,predicted_y)))
#                 closet_point=find_closest_point(xy_all,(predicted_x,predicted_y))
#                 final_pos.append([closet_point[0],closet_point[1],idx+1])
#                 x_coords[idx].append(int(closet_point[0]))
#                 if len(x_coords[idx]) > window_size:
#                     x_coords[idx].pop(0)
#                 # 保存y的序列，滑动窗口
#                 y_coords[idx].append(int(closet_point[1]))
#                 if len(y_coords[idx])>window_size:
#                     y_coords[idx].pop(0)
#             else:
#                 final_pos.append([int(ave_x),int(ave_y),idx+1])
#                 # 保存x的序列，滑动窗口
#                 x_coords[idx].append(int(ave_x))
#                 if len(x_coords[idx]) > window_size:
#                     x_coords[idx].pop(0)
#                 # 保存y的序列，滑动窗口
#                 y_coords[idx].append(int(ave_y))
#                 if len(y_coords[idx])>window_size:
#                     y_coords[idx].pop(0)
#
#         # 缺省值，直接用预测值
#         else:
#             if predicted_x:
#                 final_pos.append([int(predicted_x), int(predicted_y), idx + 1])
#
#     return final_pos

class MultiCameraFusionUtils:
    def __init__(self, max_jump_distance=50, history_size=5):
        """
        初始化多摄像头融合模块。
        - max_jump_distance: 最大允许跳跃距离，超过此值的数据将被丢弃。
        - history_size: 每个 ID 记录多少帧的历史坐标。
        """
        self.camera_positions = [(50, 550), (50, 50), (700, 50), (700, 550)]
        self.max_jump_distance = max_jump_distance
        self.last_seen = {}  # 记录 ID 上次出现的时间
        self.history_positions = defaultdict(lambda: deque(maxlen=history_size))  # 记录每个 ID 的历史位置
        self.epsilon = 1e-6  # 防止除零
        self.timeout_threshold = 30  # 超时阈值（单位：秒）
        self.idx = 0


    def vote_camera(self, camera1, camera2, camera3, camera4):
        """
        进行摄像头数据融合，剔除异常数据，并返回稳定的 ID 位置。
        """
        self.idx += 1
        all_cameras = [camera1, camera2, camera3, camera4]
        id_data = defaultdict(list)

        # 1. 收集所有摄像头的 id 坐标数据
        for cam_idx, cam_data in enumerate(all_cameras):
            for x, y, identity in cam_data:
                if len(identity) == 1:
                    if self.is_valid_position(identity, x, y):  # 过滤掉异常跳跃的数据
                        id_data[identity].append((x, y, cam_idx))
                        self.last_seen[identity] = self.idx  # 更新 ID 记录时间

        final_positions = []

        # 2. 计算融合坐标
        for identity, positions in id_data.items():
            if len(identity) == 1:
                if len(positions) == 1:
                    # 只有一个摄像头提供了该 ID，直接使用
                    # final_positions.append((positions[0][0], positions[0][1], identity))
                    continue
                else:
                    # 多个摄像头提供了该 ID，进行加权融合
                    total_weight = 0
                    weighted_x, weighted_y = 0, 0

                    for x, y, cam_idx in positions:
                        cam_x, cam_y = self.camera_positions[cam_idx]  # 摄像头坐标
                        distance = np.sqrt((x - cam_x) ** 2 + (y - cam_y) ** 2)
                        weight = 1 / (distance + self.epsilon)  # 距离越近，权重越高

                        weighted_x += x * weight
                        weighted_y += y * weight
                        total_weight += weight

                    # 计算加权平均坐标
                    fused_x = weighted_x / total_weight
                    fused_y = weighted_y / total_weight


                    # 记录历史位置
                    self.history_positions[identity].append((fused_x, fused_y))
                    final_positions.append((fused_x, fused_y, identity))

        self.cleanup_history()

        return final_positions

    def is_valid_position(self, identity, x, y):
        """
        判断当前坐标是否是合理的。
        如果新位置与最近的历史位置变化过大，则认为数据异常。
        """
        if identity not in self.history_positions or not self.history_positions[identity]:
            return True  # 该 ID 没有历史记录，直接接受

        last_x, last_y = self.history_positions[identity][-1]
        distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

        return distance <= self.max_jump_distance  # 仅接受合理变化范围内的数据

    def cleanup_history(self):
        """删除超过一定时间未出现的 ID 数据"""
        ids_to_remove = [identity for identity, last_time in self.last_seen.items()
                         if self.idx - last_time > self.timeout_threshold]

        for identity in ids_to_remove:
            if identity in self.history_positions:
                del self.history_positions[identity]
            if identity in self.last_seen:
                del self.last_seen[identity]
            # print(f"Removed stale ID {identity} from history.")

class CameraFusion:
    def __init__(self):
        self.single_camera_position = {}
        self.fused_position = []
        self.most_recent_final_position = None
        self.fusion_utils = MultiCameraFusionUtils()
        self.mismatch_dict = {'camera1': 0,'camera2':0,'camera3':0,'camera4':0}

    def update_single_camera_position(self,camera_name,data):
        self.single_camera_position[camera_name] = data

    def fuse_position(self):
        self.most_recent_final_position = self.fusion_utils.vote_camera(
                                self.single_camera_position["camera1"],
                                self.single_camera_position["camera2"],
                                self.single_camera_position["camera3"],
                                self.single_camera_position["camera4"])
        return self.most_recent_final_position

    def find_difference_and_feedback(self, tracking_results, reset_queue,fix_queue,camera_name, threshold=100):
        if self.most_recent_final_position is None:
            return  # 如果没有最终位置数据，则不做处理
        tracking_results = tracking_results.copy()
        # 构建一个字典，便于查找最终位置数据
        final_position_dict = {f"{entry[2]}": (entry[0], entry[1]) for entry in self.most_recent_final_position}
        unmatched_track = []

        for track in tracking_results:
            x, y, identity = track
            if len(identity) == 1:
                if identity in final_position_dict:
                    final_x, final_y = final_position_dict[identity]
                    # 计算欧几里得距离
                    distance = np.sqrt((x - final_x) ** 2 + (y - final_y) ** 2)

                    if distance > threshold:
                        if self.mismatch_dict[camera_name] < 15:
                            self.mismatch_dict[camera_name] += 1
                        else:
                            reset_queue.put(identity)  # 记录偏离较远的ID信息
                            self.mismatch_dict[camera_name] = 0

                    del final_position_dict[identity]
                else:
                    unmatched_track.append(track)

        for track in unmatched_track:
            x,y,identity = track
            fix_queue.put({identity:-1})