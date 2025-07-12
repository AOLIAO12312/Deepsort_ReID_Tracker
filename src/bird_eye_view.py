import cv2
import numpy as np
import math
from sklearn.linear_model import LinearRegression

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


    def bbox2coord(self,positions:list):
        # 测试中
        positions = np.array([positions], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(positions, self.matrix)
        return transformed_point[0]

    def draw_bird_view(self,tracking_results,eye_position):
        positions = []
        for tracking_result in tracking_results:
            bbox = tracking_result[:4]
            cx = (bbox[0] + bbox[2]) / 2
            # cy = bbox[3] - (bbox[3] - bbox[1]) / 15
            cy = bbox[3]
            positions.append([cx, cy])
            # eye_position.append([cx,cy,tracking_result[4]])
            # print(cx,cy,tracking_result[4])
        if len(positions) != 0:
            # 第一版
            # positions = np.array([positions], dtype=np.float32)
            # transformed_point = cv2.perspectiveTransform(positions, self.matrix)

            # 第二版：测试中
            transformed_point = self.bbox2coord(positions)
            # print(transformed_point)
            field = self.field.copy()
            for i,point in enumerate(transformed_point):
                player_x, player_y = point[0], point[1]
                mapped_x = int(player_x + 50)
                mapped_y = int(player_y + 50)
                # print(f"第{i}次循环")
                eye_position.append([mapped_x,mapped_y,tracking_results[i][4]])
                cv2.circle(field, (mapped_x, mapped_y), 7, (0, 0, 255), -1)
                cv2.putText(field, str(tracking_results[i][4]), (mapped_x + 10, mapped_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return field
        else:
            return self.field.copy()

    def draw_final_pos_view(self, final_pos):
        """
        绘制融合后的最终位置，并在图像上标注每个 ID
        """
        if final_pos:
            field = self.field.copy()  # 复制原始场地图像

            for point in final_pos:
                mapped_x = int(point[0])  # x 坐标
                mapped_y = int(point[1])  # y 坐标
                color = (0, 0, 255)  # 默认蓝色

                # 根据 ID 判断颜色
                if point[2] in ["1", "2", "3", "4"]:  # 假设 ID 1-4 用蓝色
                    color = (0, 0, 255)  # 蓝色
                else:  # 其他 ID 用红色
                    color = (255, 0, 0)  # 红色

                # 绘制圆圈标记位置
                cv2.circle(field, (mapped_x, mapped_y), 7, color, -1)

                # 绘制 ID 文字
                cv2.putText(field, str(point[2]), (mapped_x + 10, mapped_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            return field
        else:
            return self.field.copy()  # 如果没有数据，返回原始场地图像

    # def vote_camera(self,list1,list2,list3,list4,x_coords,y_coords):
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
    #
    #
    #     # print('final_pos:', final_pos)
    #
    #     if final_pos:
    #         field = self.field.copy()
    #         for i, point in enumerate(final_pos):
    #             mapped_x = point[0]
    #             mapped_y = point[1]
    #             color=(0,0,255)
    #             if point[2] in [1,2,3,4]:
    #                 color=(0,0,255)
    #             else:
    #                 color=(255,0,0)
    #             cv2.circle(field, (mapped_x, mapped_y), 7, color, -1)
    #             cv2.putText(field, str(point[2]), (mapped_x + 10, mapped_y - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    #
    #         return field
    #     else:
    #         return self.field.copy()
