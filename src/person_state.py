from enum import Enum

class GameState(Enum):
    DEFEND = 0
    ATTACK = 1
    NOTIN = 2

class PersonState:
    def __init__(self, person_id):
        # 运动员的id编号
        self.id = person_id
        # 运动员的当前位置
        self.position = []
        # 运动员的当前速度
        self.velocity = float(0)
        # 运动员的当前加速度
        self.acceleration = float(0)
        # 运动员运动的方向
        self.direction = float(0)

        # 运动员的当前状态（防守、进攻）（依据运动员的位置判定）
        self.state = GameState.DEFEND
