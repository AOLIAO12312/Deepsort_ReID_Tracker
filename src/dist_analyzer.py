import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np
from scipy.interpolate import make_interp_spline
# === 修改为你的文件路径 ===
csv_file_path = 'C:\\Users\\asus\\Desktop\\data.csv'

# 数据结构：key = (attacker_id, defender_id), value = list of (frame_idx, pose_dist/coord_dist)
pose_dists = defaultdict(list)
coord_dists = defaultdict(list)

# 读取数据
with open(csv_file_path, 'r') as f:
    reader = csv.DictReader(f)
    for frame_idx, row in enumerate(reader):
        attacker_id = int(row['attacker_id'].strip())
        defender_id = int(row['defender_id'].strip())
        pose_min_dist = float(row['pose_min_dist'])
        coord_dist = float(row['coord_dist'])

        key = (attacker_id, defender_id)
        pose_dists[key].append((frame_idx, pose_min_dist))
        coord_dists[key].append((frame_idx, coord_dist))

# 平滑函数：使用 B样条插值
def smooth_curve(x, y, smooth_points=300):
    if len(x) < 4:
        return x, y  # 不足以拟合样条，返回原始
    x = np.array(x)
    y = np.array(y)
    x_new = np.linspace(x.min(), x.max(), smooth_points)
    spline = make_interp_spline(x, y, k=3)  # k=3 为立方样条
    y_smooth = spline(x_new)
    return x_new, y_smooth

# 绘图
plt.figure(figsize=(14, 6))

# --- pose_min_dist 平滑曲线 ---
plt.subplot(1, 2, 1)
for (atk_id, def_id), data in pose_dists.items():
    frames, values = zip(*data)
    x_smooth, y_smooth = smooth_curve(frames, values)
    plt.plot(x_smooth, y_smooth, label=f'A:{atk_id}, D:{def_id}')
plt.title('Pose Min Distance (Smoothed)')
plt.xlabel('Frame Index')
plt.ylabel('Pose Min Distance')
plt.legend()

# --- coord_dist 平滑曲线 ---
plt.subplot(1, 2, 2)
for (atk_id, def_id), data in coord_dists.items():
    frames, values = zip(*data)
    x_smooth, y_smooth = smooth_curve(frames, values)
    plt.plot(x_smooth, y_smooth, label=f'A:{atk_id}, D:{def_id}')
plt.title('Coordinate Distance (Smoothed)')
plt.xlabel('Frame Index')
plt.ylabel('Coordinate Distance')
plt.legend()

plt.tight_layout()
plt.show()
