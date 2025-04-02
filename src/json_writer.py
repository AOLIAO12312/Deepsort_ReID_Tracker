import json


class JsonWriter:

    def __init__(self,output_file):
        self.all_frames = []
        self.output_file = output_file

    def update(self,mapped_results,idx):
        MAX_FRAMES = 4000
        # 获取队列中的一帧数据

        # 解析当前帧数据
        frame_data = {
            "frame_id": idx,
            "person_data": []
        }

        for bbox in mapped_results:
            x_min, y_min, x_max, y_max, obj_id = bbox
            if isinstance(obj_id, str) and obj_id.startswith("Unknown"):
                obj_id = -1  # 设置 Unknown ID 为 -1

            frame_data["person_data"].append({
                "id": obj_id,
                "bbox": [x_min, y_min, x_max, y_max]
            })
            # **确保所有数值类型可 JSON 序列化**
            for obj in frame_data["person_data"]:
                obj["id"] = int(obj["id"])  # 转换 np.int32 为 Python int
                obj["bbox"] = [int(x) for x in obj["bbox"]]  # 转换 numpy 数据类型
        self.all_frames.append(frame_data)

        # 定期写入文件（比如每500帧写一次）
        if len(self.all_frames) % 100 == 0:
            with open(self.output_file, "w") as f:
                json.dump({"frames": self.all_frames}, f, indent=4)
            print(f"已保存 {len(self.all_frames)} 帧数据到 {self.output_file}")

        # 结束条件：如果存储超过 MAX_FRAMES，退出线程
        if len(self.all_frames) >= MAX_FRAMES:
            print("达到最大帧数，停止写入。")
            return
