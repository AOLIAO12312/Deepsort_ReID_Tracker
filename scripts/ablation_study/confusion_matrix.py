import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_json(file_path):
    """加载 JSON 数据"""
    import json
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def iou(box1, box2):
    """计算两个边界框的 IoU（交并比）"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1, yi1, xi2, yi2 = max(x1, x1g), max(y1, y1g), min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def match_boxes(gt_boxes, test_boxes, iou_threshold=0.65):
    """匹配标准框和测试框，基于 IoU 选择重合度最高的对"""
    matches = {}
    used_gt = set()
    used_test = set()

    for gt_box in gt_boxes:
        best_match = None
        best_iou = iou_threshold
        for test_box in test_boxes:
            if test_box in used_test:
                continue
            iou_score = iou(gt_box, test_box)
            if iou_score > best_iou:
                best_match = test_box
                best_iou = iou_score

        if best_match:
            matches[gt_box] = best_match
            used_gt.add(gt_box)
            used_test.add(best_match)
        else:
            matches[gt_box] = None
            used_gt.add(gt_box)

    return matches

def process_data(gt_data, test_data, num_classes=8):
    """处理单个摄像头的数据，提取 y_true 和 y_pred"""
    y_true = []
    y_pred = []

    for gt_frame, test_frame in zip(gt_data["frames"], test_data["frames"]):
        gt_persons = {tuple(person["bbox"]): person["id"] for person in gt_frame["person_data"]}
        test_persons = {tuple(person["bbox"]): person["id"] for person in test_frame["person_data"]}

        matches = match_boxes(gt_persons.keys(), test_persons.keys())

        for gt_bbox, test_bbox in matches.items():
            if test_bbox is None:
                continue  # 跳过未匹配的情况
            gt_id = gt_persons[gt_bbox]
            test_id = test_persons[test_bbox]

            if gt_id in range(1, num_classes + 1) and test_id in range(1, num_classes + 1):
                y_true.append(gt_id)
                y_pred.append(test_id)

    return y_true, y_pred


def plot_combined_confusion_matrix(gt_files, test_files, num_classes=8):
    """计算多个摄像头的混淆矩阵并绘制"""
    y_true_all = []
    y_pred_all = []

    for gt_file, test_file in zip(gt_files, test_files):
        gt_data = load_json(gt_file)
        test_data = load_json(test_file)
        y_true, y_pred = process_data(gt_data, test_data, num_classes)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true_all, y_pred_all, labels=list(range(1, num_classes + 1)))

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(1, num_classes + 1),
                yticklabels=range(1, num_classes + 1))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Combined Confusion Matrix for All Cameras")
    plt.show()

def analyze_reid_correction(gt_files, test_files_full_model, test_files_no_fusion):
    """分析融合模块对 ReID 错误的修正情况（支持多个文件）"""
    reid_error_count = 0
    reid_corrected_count = 0

    # 遍历所有文件
    for gt_file, full_file, reid_file in zip(gt_files, test_files_full_model, test_files_no_fusion):
        gt_data = load_json(gt_file)
        full_data = load_json(full_file)
        reid_data = load_json(reid_file)

        for gt_frame, full_frame, reid_frame in zip(gt_data["frames"], full_data["frames"], reid_data["frames"]):
            gt_persons = {tuple(person["bbox"]): person["id"] for person in gt_frame["person_data"]}
            full_persons = {tuple(person["bbox"]): person["id"] for person in full_frame["person_data"]}
            reid_persons = {tuple(person["bbox"]): person["id"] for person in reid_frame["person_data"]}

            # 计算 ReID 错误
            for bbox in gt_persons:
                gt_id = gt_persons[bbox]
                if gt_id == -1:
                    continue
                reid_id = reid_persons.get(bbox, -1)
                full_id = full_persons.get(bbox, -1)

                if reid_id != gt_id:  # 如果 ReID 预测错误
                    reid_error_count += 1
                    if full_id == gt_id:  # 但 Full Model 修正了它
                        reid_corrected_count += 1

    correction_rate = (reid_corrected_count / reid_error_count * 100) if reid_error_count > 0 else 0

    print(f"处理的摄像头数量: {len(gt_files)}")
    print(f"ReID 误识别总次数: {reid_error_count}")
    print(f"融合模块修正次数: {reid_corrected_count}")
    print(f"修正率: {correction_rate:.2f}%")

if __name__ == "__main__":
    gt_files = [
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\benchmark_data\\camera1_basedata.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\benchmark_data\\camera2_basedata.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\benchmark_data\\camera3_basedata.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\benchmark_data\\camera4_basedata.json"
    ]


    test_files_full_model = [
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\Full_Model\\camera1_full_model.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\Full_Model\\camera2_full_model.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\Full_Model\\camera3_full_model.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\Full_Model\\camera4_full_model.json"
    ]

    test_files_no_fusion = [
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\No_Filter\\camera1_no_filter.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\No_Filter\\camera1_no_filter.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\No_Filter\\camera1_no_filter.json",
        "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\No_Filter\\camera1_no_filter.json"
    ]


    analyze_reid_correction(gt_files,test_files_full_model,test_files_no_fusion)


    # test_files = [
    #     "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\Only_ReID\\camera1_test_only_ReID.json",
    #     "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\Only_ReID\\camera2_test_only_ReID.json",
    #     "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\Only_ReID\\camera3_test_only_ReID.json",
    #     "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\Only_ReID\\camera4_test_only_ReID.json"
    # ]
    # plot_combined_confusion_matrix(gt_files, test_files)

