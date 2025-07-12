import csv
import os
from collections import deque
import math
from src.person_state import PersonState, GameState


# 用于解析场景状态及运动员相关信息
def save_person_state_to_csv(persons_state, frame_idx, filepath, first_write=False):
    mode = 'w' if first_write else 'a'  # 首次写入用覆盖
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode=mode, newline='') as f:
        writer = csv.writer(f)
        if first_write or not file_exists:
            writer.writerow([
                "frame_idx", "person_id", "x", "y",
                "velocity", "acceleration", "direction", "state"
            ])

        for person in persons_state:
            x, y = person.position if person.position else (None, None)
            writer.writerow([
                frame_idx,
                person.id,
                round(x, 3) if x else None,
                round(y, 3) if y else None,
                round(person.velocity, 3),
                round(person.acceleration, 3) if person.acceleration is not None else None,
                round(person.direction, 1),
                person.state.name
            ])


class StateAnalyzer:
    def __init__(self):
        # 日志文件写入
        self.first_write = True

        self.fps = 30
        self.frame_interval = 15  # 取0.5秒
        # 保存3s内的运动员位置数据
        self.history_position = deque(maxlen=90)
        self.persons_state = []

        # 插入1-8的运动员编号
        for i in range(1, 9):
            self.persons_state.append(PersonState(str(i)))

    def update(self,final_position):
        # 转换为字典方便使用
        final_position_dict = {item[2]: (item[0], item[1]) for item in final_position}

        # 将像素坐标修正为实际物理坐标
        adjusted_position_dict = {
            pid: ((x - 50) / 50, (y - 50) / 50)
            for pid, (x, y) in final_position_dict.items()
        }

        self.history_position.append(adjusted_position_dict)

        if len(self.history_position) >= self.frame_interval * 2 + 1:
            self.compute_motion_info()

    def compute_motion_info(self):
        teamA = {'1','2','3','4'}
        teamB = {'5','6','7','8'}
        dt = self.frame_interval / self.fps  # Δt = 0.5秒

        # 提取 frame_t_minus_30, frame_t_minus_15, frame_t
        frames = list(self.history_position)
        frame_t_minus_30 = frames[-(self.frame_interval * 2 + 1)]
        frame_t_minus_15 = frames[-(self.frame_interval + 1)]
        frame_t = frames[-1]

        for person in self.persons_state:
            pid = person.id
            # 先判断是否存在必要帧
            if pid in frame_t and pid in frame_t_minus_15:
                x1, y1 = frame_t_minus_15[pid]
                x2, y2 = frame_t[pid]

                person.position = (x2, y2)

                if (person.id in teamA and x2 > 6.5) or (person.id in teamB and x2 < 6.5):
                    person.state = GameState.ATTACK
                else:
                    person.state = GameState.DEFEND

                dx = x2 - x1
                dy = y2 - y1
                v_now = math.sqrt(dx ** 2 + dy ** 2) / dt
                direction = math.degrees(math.atan2(dy, dx))

                person.velocity = v_now
                person.direction = direction

                # 如果还能拿到 t-30 帧，计算加速度
                if pid in frame_t_minus_30:
                    x0, y0 = frame_t_minus_30[pid]
                    dx_prev = x1 - x0
                    dy_prev = y1 - y0
                    v_prev = math.sqrt(dx_prev ** 2 + dy_prev ** 2) / dt
                    acceleration = (v_now - v_prev) / dt
                    person.acceleration = acceleration
                else:
                    person.acceleration = float(0)
            else:
                # 未找到则代表运动员不在场内
                person.state = GameState.NOTIN

    def get_persons_state(self):
        return self.persons_state

