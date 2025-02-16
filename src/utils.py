import cv2
from models.deep_sort_pytorch.utils.draw import palette
import torch.nn.functional as F

def yolo2coordinates(x, y, w, h, img_w, img_h):
    xmin = round(img_w * (x - w / 2.0))
    xmax = round(img_w * (x + w / 2.0))
    ymin = round(img_h * (y - h / 2.0))
    ymax = round(img_h * (y + h / 2.0))
    return xmin, ymin, xmax, ymax

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0], xyxy[2]])
    bbox_top = min([xyxy[1], xyxy[3]])
    bbox_w = abs(xyxy[0] - xyxy[2])
    bbox_h = abs(xyxy[1] - xyxy[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    """
    Draw bbox and id
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def draw_keypoints(img,keypoints,confidence_threshold):
    keypoints = keypoints.cpu().numpy()
    for i in range(keypoints.shape[0]):
        keypoint = keypoints[i]
        filtered_keypoint = []
        for individual_point in keypoint:
            if individual_point[2] > confidence_threshold:
                filtered_keypoint.append(individual_point)

        for kp in filtered_keypoint:
            x, y = kp[:2]  # 获取 (x, y) 坐标
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    return img

def normalize_feature(feature):
    return F.normalize(feature, p=2, dim=1)

def compute_iou(box1, box2):
    """
    Calculate IoU of two boxes
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_bbox_by_id(tracking_results, target_id):
    for result in tracking_results:
        if result[-1] == target_id:
            return result[:4]
    return None

def backtrack_match(distances, current_row, selected_persons, current_distance, best_result):
    if current_row == len(distances):
        if current_distance < best_result[0]:
            best_result[0] = current_distance
            best_result[1] = selected_persons.copy()
        return
    candidates = distances[current_row]
    for person, dist in candidates:
        if person not in selected_persons:
            selected_persons.append(person)
            backtrack_match(distances, current_row + 1, selected_persons, current_distance + dist, best_result)
            selected_persons.remove(person)


def match_photos_to_persons(distances):
    best_result = [float('inf'), []]
    backtrack_match(distances, 0, [], 0, best_result)
    return best_result[1], best_result[0]

def get_border(frame):
    """
    Initialize the boundary and return the border
    """
    points = []
    done = False
    print("Initializing the boundary, please click on the 4 corners of the boundary...")

    def click_event(event, x, y, flags, params):
        nonlocal done
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            points.append((x, y))
            print(f"Point clicked: ({x}, {y})")
            cv2.imshow("Boundary select", frame)
            if len(points) == 4:
                print("Four points selected: ", points)
                done = True
    cv2.imshow("Boundary select", frame)
    cv2.setMouseCallback("Boundary select", click_event)
    while not done:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    print("Final points:", points)
    return points

def draw_reid_tracking_results(results,frame):
    for result in results:
        bbox = result[:4]
        name = result[4]
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 0, 0)
        thickness = 2
        label = f"ID: {name}"
        frame = cv2.putText(frame, label, (x1, y2 + 20), font, font_scale, font_color, thickness)