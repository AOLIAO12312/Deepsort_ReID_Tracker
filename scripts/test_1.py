import math
import queue
import threading
from src.person_database import PersonDatabase
import cv2
import glob
from src.bounding_box_filter import BoundingBoxFilter
from models.yolo.yolo_detector import YoloDetector
from models.deep_sort_pytorch.deepsort_tracker import DeepsortTracker
from src.utils import xyxy_to_xywh
import torch

bound1 = [(709, 122),(1124, 196),(729, 702),(145, 289)]
bound2 = [(249, 165),(638, 95),(1193, 268),(500, 708)]
bound3 = [(707, 66),(1135, 146),(752, 683),(140, 245)]
bound4 = [(149, 247),(541, 154),(1083, 281),(586, 714)]
bound = [bound1,bound2,bound3,bound4]
bounding_box_filter = BoundingBoxFilter(bound4,0.1,0.5)

person_database = PersonDatabase()
deepsort_tracker = DeepsortTracker("/configs/deep_sort.yaml", 'cpu')
deepsort = deepsort_tracker.get_tracker()
detector = YoloDetector("/models/yolo/weights/yolo11n.pt", 'cpu')

for i in range(8):
    folder_path = f"/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/input/Athlete/{i + 1}/*.png"
    image_paths = sorted(glob.glob(folder_path))
    images = [cv2.imread(img_path) for img_path in image_paths]
    person_database.add_person(f'Person{i+1}',images)


def backtrack_match(distances, current_row, selected_persons, current_distance, best_result):
    if current_row == len(distances):
        if current_distance < best_result[0]:
            best_result[0] = current_distance
            best_result[1] = selected_persons.copy()
        return
    candidates = distances[current_row]
    for person, dist in candidates:
        if person not in selected_persons:
            # 选择当前person并继续回溯
            selected_persons.append(person)
            backtrack_match(distances, current_row + 1, selected_persons, current_distance + dist, best_result)
            # 回溯: 撤销选择
            selected_persons.remove(person)


def match_photos_to_persons(distances):
    best_result = [float('inf'), []]  # 最优结果（最小L2距离和，最优匹配的persons）
    backtrack_match(distances, 0, [], 0, best_result)
    return best_result[1], best_result[0]  # 返回匹配结果及其对应的最小L2距离和

# 记录当前处理的帧数
frame_counter = 0
# DeepSORT ID → 运动员 ID 映射表
deepsort_to_athlete = {}  # {deepsort_id: athlete_id}
# DeepSORT 丢失 ID 计数
id_lost_count = {}  # {deepsort_id: 丢失的帧数}
LOST_THRESHOLD = 100

reset_queue = queue.Queue()

def wait_for_input():
    while True:
        user_input = input("请输入需要重置的人员（按回车确认）：")
        reset_queue.put(user_input)  # 将输入的数据放入队列中

def identify(cropped_images):
    data = []
    distances = []
    deleted_idx = []
    for cropped_image in cropped_images:
        results = person_database.search(cropped_image, len(cropped_images))
        data.append(results)

    while len(data) > len(person_database.database):
        max_index = max(range(len(data)), key=lambda j: data[j][0][1])
        deleted_idx.append(max_index)
        del data[max_index]
    matching_persons, min_distance = match_photos_to_persons(data)

    for i, results in enumerate(data):
        for result in results:
            if result[0] == matching_persons[i]:
                distances.append(result[1])
                break

    for i, idx in enumerate(deleted_idx):
        if idx < len(matching_persons):
            matching_persons.insert(idx, None)
            distances.insert(idx, math.inf)
        else:
            matching_persons.append(None)
            distances.append(math.inf)

    return matching_persons, distances

def get_bbox_by_id(tracking_results, target_id):
    for result in tracking_results:
        if result[-1] == target_id:
            return result[:4]
    return None

def map_deepsort_to_athlete(tracking_results, orig_img,frame_idx):
    """ 将 DeepSORT ID 映射到实际运动员身份 ID """
    global deepsort_to_athlete, id_lost_count
    mapped_results = []
    unassigned_tracks = []  # 存储未分配身份的 tracks
    unassigned_deepsort_ids = []  # 存储对应的 DeepSORT ID
    update_tracks_names = []
    update_tracks_image = []
    active_ids = set()  # 记录当前活跃的 DeepSORT ID
    for track in tracking_results:
        deepsort_id = track[4]
        bbox = track[:4]
        active_ids.add(deepsort_id)
        # 如果这个 ID 还没有对应的运动员身份，加入未分配列表
        if deepsort_id not in deepsort_to_athlete:
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = orig_img[y1:y2, x1:x2]
            unassigned_tracks.append(cropped_image)
            unassigned_deepsort_ids.append(deepsort_id)
        else:
            if frame_counter % 30 == 0:
                x1, y1, x2, y2 = map(int, bbox)
                cropped_image = orig_img[y1:y2, x1:x2]
                update_tracks_image.append(cropped_image)
                update_tracks_names.append(deepsort_to_athlete[deepsort_id])

    if frame_counter % 30 == 0:
        if not reset_queue.empty():
            user_input = reset_queue.get()
            print(f"等待{user_input}信息被重置...")
            person_database.update_person_feature_and_rebuild_index(update_tracks_names,update_tracks_image,3,[user_input])
            existing_deepsort_ids = [k for k, v in deepsort_to_athlete.items() if v == user_input]
            if len(existing_deepsort_ids) > 0:
                del deepsort_to_athlete[existing_deepsort_ids[0]]
        else:
            person_database.update_person_feature_and_rebuild_index(update_tracks_names, update_tracks_image, 3,
                                                                    [])

    matching_persons,distances = identify(unassigned_tracks)
    # 解决相同运动员对应多个 DeepSORT ID 的情况
    for idx, deepsort_id in enumerate(unassigned_deepsort_ids):
        athlete_id = matching_persons[idx]
        distance = distances[idx]
        if distance < 0.38: # L2距离低于阈值则可以认为该辨别置信度足够高，可以分配ID
            if athlete_id in deepsort_to_athlete.values():
                # 如果运动员已经有ID对应，但当前的 DeepSORT ID 没有分配身份
                existing_deepsort_ids = [k for k, v in deepsort_to_athlete.items() if v == athlete_id]
                for existing_id in existing_deepsort_ids:
                    # 如果已有的 ID 与当前的 ID 不同，则进行决斗
                    existing_bbox = get_bbox_by_id(tracking_results,existing_id)
                    current_bbox = get_bbox_by_id(tracking_results,deepsort_id)
                    if existing_bbox is None:
                        deepsort_to_athlete[deepsort_id] = athlete_id # 若未找到则直接指定
                        del deepsort_to_athlete[existing_id]  # 踢出已绑定的ID
                        break
                    # 提取两个候选框，进行决斗
                    existing_cropped = orig_img[existing_bbox[1]:existing_bbox[3], existing_bbox[0]:existing_bbox[2]]
                    current_cropped = orig_img[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2]]

                    # 通过 identify 方法比较两者
                    matching_persons_tmp, distances_tmp = identify([existing_cropped, current_cropped])
                    if matching_persons_tmp[1] == athlete_id:
                        # 决斗成功，当前 ID 获得该运动员身份
                        deepsort_to_athlete[deepsort_id] = athlete_id
                        del deepsort_to_athlete[existing_id]  # 踢出已绑定的ID
                        break
                    else:
                        break # 不执行操作
            else:
                deepsort_to_athlete[deepsort_id] = athlete_id

    # 将结果加入最终映射
    for track in tracking_results:
        deepsort_id = track[4]
        if deepsort_id in deepsort_to_athlete:
            mapped_results.append([*track[:4], deepsort_to_athlete[deepsort_id]])
        else:
            mapped_results.append([*track[:4], f'Unknown {deepsort_id}'])  # 未分配的情况
    handle_lost_ids(active_ids) # 维护丢失的ID
    return mapped_results

def handle_lost_ids(active_ids):
    global id_lost_count
    lost_ids = set(deepsort_to_athlete.keys()) - active_ids
    for lost_id in lost_ids:
        id_lost_count[lost_id] = id_lost_count.get(lost_id, 0) + 1
        # 如果 DeepSORT ID 丢失太久，移除映射
        if id_lost_count[lost_id] >= LOST_THRESHOLD:
            del deepsort_to_athlete[lost_id]
            del id_lost_count[lost_id]



video_capture = cv2.VideoCapture('/data/input/long_video/Camera4.mp4')
input_thread = threading.Thread(target=wait_for_input, daemon=True)
input_thread.start()


outputs = []
bbox_xyxy =[]
identities = []
while True:
    ret,frame = video_capture.read()
    if not ret:
        break  # 如果读取不到帧，则退出
    result = detector.get_result(frame)
    frame,xyxy,conf = bounding_box_filter.box_filter(frame,result)
    if xyxy is not None:
        xywhs = torch.empty(0, 4)
        confss = torch.empty(0, 1)
        # 此处对接lhx筛选出的关键人物部分
        # 对每个人物检测结果绘制边界框
        for i, (bbox, confidence) in enumerate(zip(xyxy, conf)):
            x1, y1, x2, y2 = map(int, bbox)
            x_c, y_c, w, h = xyxy_to_xywh(x1, y1, x2, y2)
            xywhs = torch.cat((xywhs, torch.tensor([x_c, y_c, w, h]).unsqueeze(0)), dim=0)
            confss = torch.cat((confss, torch.tensor([confidence]).unsqueeze(0)), dim=0)
        # 进行人体的持续跟踪
        outputs = deepsort.update(xywhs, confss, frame)
    else:
        deepsort.increment_ages()
    frame_counter += 1
    mapped_results = map_deepsort_to_athlete(outputs, frame, frame_counter)
    for mapped_result in mapped_results:
        bbox = mapped_result[:4]
        id = mapped_result[4]
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0)  # 绿色框
        thickness = 2  # 边框线宽
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 在边框下方标注 ID
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        font_scale = 0.7  # 字体大小
        font_color = (255, 0, 0)  # 红色字体
        thickness = 2  # 字体粗细
        label = f"ID: {id}"
        # 将文本放在框下方，y坐标设置为 y2 + 20
        frame = cv2.putText(frame, label, (x1, y2 + 20), font, font_scale, font_color, thickness)
    cv2.imshow("Monitor", frame)
    cv2.waitKey(1)

# 释放资源
video_capture.release()
cv2.destroyAllWindows()


