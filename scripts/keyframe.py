import matplotlib.pyplot as plt

# 数据
total_frames = 3400
key_frames = 188
actual_collisions = 35
non_collision_frames = key_frames - actual_collisions

# 饼图：关键帧中 碰撞 vs 非碰撞
plt.figure(figsize=(6, 6))
plt.pie([actual_collisions, non_collision_frames],
        labels=['碰撞关键帧', '非碰撞关键帧'],
        autopct='%1.1f%%',
        colors=['#ff9999','#66b3ff'])
plt.title('关键帧中碰撞与非碰撞占比')
plt.show()

# 柱状图：进攻场景 vs 成功提取
attack_total = 6
attack_success = 5

plt.figure(figsize=(6, 4))
plt.bar(['总进攻次数', '成功提取场景数'], [attack_total, attack_success], color=['#ffcc99', '#99ff99'])
plt.title('关键帧提取成功率')
plt.ylim(0, 7)
plt.ylabel('次数')
plt.show()

# 条形图：每个进攻场景提取帧数范围（平均30-40）
scene_ids = [f'进攻{i+1}' for i in range(5)]
frame_counts = [35, 32, 38, 31, 34]  # 示例数据，取均值范围内的值模拟

plt.figure(figsize=(8, 4))
plt.bar(scene_ids, frame_counts, color='#c2c2f0')
plt.title('每个进攻场景提取的关键帧数量')
plt.ylabel('关键帧数')
plt.ylim(0, 50)
plt.axhline(30, color='gray', linestyle='--', linewidth=1, label='下限 30')
plt.axhline(40, color='gray', linestyle='--', linewidth=1, label='上限 40')
plt.legend()
plt.show()
