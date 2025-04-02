import json

from tqdm import tqdm


def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r') as f:
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


def calculate_id_metrics(gt_data, test_data):
    """计算 ID 准确度、ID 切换率、MOTA、MOTP 等"""
    correct_id_count = 0
    total_id_count = 0
    previous_id_map = {}
    correct_id_rate_count = 0
    total_test_id_count = 0
    fn_count = 0  # False Negatives（漏检数）
    fp_count = 0  # False Positives（误检数）
    iou_total = 0
    matched_count = 0
    IDTP = 0
    IDFP = 0
    IDFN = 0

    for gt_frame, test_frame in zip(gt_data["frames"], test_data["frames"]):
        gt_persons = {tuple(person["bbox"]): person["id"] for person in gt_frame["person_data"]}
        test_persons = {tuple(person["bbox"]): person["id"] for person in test_frame["person_data"]}

        matches = match_boxes(gt_persons.keys(), test_persons.keys())

        unmatch = 0
        for gt_bbox, test_bbox in matches.items():
            if test_bbox is None:
                unmatch += 1
                IDFN += 1
                continue
            gt_id = gt_persons[gt_bbox]
            test_id = test_persons[test_bbox]
            total_id_count += 1
            iou_total += iou(gt_bbox, test_bbox)
            matched_count += 1

            if gt_id != -1:
                if gt_id == test_id:
                    IDTP += 1
                    correct_id_count += 1
                else:
                    IDFP += 1
            else:
                total_id_count -= 1


        # 计算 FN（GT 中未匹配上的目标）
        fn_count += len(gt_persons) - len(matches) + unmatch

        # 计算 FP（TEST 中未匹配上的目标）
        fp_count += len(test_persons) - len(matches) + unmatch

        for test_id in test_persons.values():
            total_test_id_count += 1
            if test_id in gt_persons.values():
                correct_id_rate_count += 1

    id_accuracy = correct_id_count / total_id_count if total_id_count > 0 else 0
    mota = 1 - ((fn_count + fp_count + (total_id_count - correct_id_count)) / total_id_count)
    motp = iou_total / matched_count if matched_count > 0 else 0
    IDF1 = (2 * IDTP) / (2 * IDTP + IDFP + IDFN)
    Rcll = IDTP/(IDTP + IDFN)
    Prcn = IDTP/(IDTP + IDFP)

    return id_accuracy, mota, motp,IDF1,Rcll,Prcn




if __name__ == "__main__":
    gt_file = "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\benchmark_data\\camera1_basedata.json"  # 标准文件
    test_file = "E:\\Deepsort_ReID_Tracker\\scripts\\ablation_study\\test_data\\No_Fusion\\camera1_test_no_fusion.json"  # 测试文件

    gt_data = load_json(gt_file)
    test_data = load_json(test_file)

    id_accuracy,mota, motp,IDF1,Rcll,Prcn = calculate_id_metrics(gt_data, test_data)

    print(f"MOTA: {mota:.2%}")
    print(f"MOTP: {motp:.2%}")
    print(f"ID Accuracy: {id_accuracy:.2%}")
    print(f"Rcll: {Rcll:.2%}")
    print(f"Prcn: {Prcn:.2%}")
    print(f"IDF1: {IDF1:.2%}")
