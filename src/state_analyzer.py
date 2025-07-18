import csv
import os
from collections import deque,Counter
import math
from copy import deepcopy

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

        # 运动员当前的详细信息
        self.persons_state = []

        # 运动员的历史信息（保留历史数据）
        self.history_state = deque(maxlen=90)  # 每帧一个{person_id: GameState}

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
        # 保留历史信息
        self.history_state.append(deepcopy(self.persons_state))

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

    def get_current_attacker(self):
        """
        Analyze the most likely current attacker based on recent history (last 0.5s).
        Returns attacker_id (str) or None if no attacker is clearly present.
        """
        if len(self.history_state) < self.frame_interval:
            return None  # Not enough data

        # 取最近 frame_interval 帧的数据
        recent_states = list(self.history_state)[-self.frame_interval:]

        # 统计每个人在这段时间内被认为是 ATTACK 的次数
        attack_counter = Counter()
        for frame_state in recent_states:
            for person_state in frame_state:
                if person_state.state == GameState.ATTACK:
                    attack_counter[person_state.id] += 1

        if not attack_counter:
            return None  # No attacker found

        # 找到出现次数最多的
        most_common = attack_counter.most_common()
        top_pid, top_count = most_common[0]

        # 如果有并列，可以用“连续出现的帧数”或“速度”进一步判断（此处略去）
        # 比如再检查 top_pid 是否最近连续3帧都为ATTACK

        return top_pid

