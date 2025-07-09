import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 设置绘图风格
sns.set(style="whitegrid")

# 数据准备
data = {
    "Experiment": ["No Filter", "No DeepSORT", "No Fusion", "Full Model"],
    "MOTA": [41.26, 54.09, 62.41, 66.12],
    "IDF1": [82.24, 86.63, 88.49, 90.26],
    "Precision": [71.79, 78.21, 82.12, 84.23],
    "Recall": [96.79, 97.10, 97.28, 97.29],
    "Speed (fps)": [3.42, 1.87, 9.28, 9.14]
}

df = pd.DataFrame(data)

# 融合用于左轴柱状图的指标
metrics = ["MOTA", "IDF1", "Precision", "Recall"]
df_melted = df.melt(id_vars=["Experiment", "Speed (fps)"],
                    value_vars=metrics,
                    var_name="Metric",
                    value_name="Value")

# 创建图像和主坐标轴
fig, ax1 = plt.subplots(figsize=(12, 6))

# 左轴：性能指标柱状图
barplot = sns.barplot(data=df_melted, x="Experiment", y="Value", hue="Metric", ax=ax1)
ax1.set_ylabel("Score (%)", fontsize=12)
ax1.set_ylim(40, 100)
ax1.set_title("Performance and Speed Across Ablation Settings", fontsize=14)

# ——给柱状图添加数据标签——
for container in barplot.containers:
    barplot.bar_label(container, fmt='%.2f', label_type='edge', padding=2, fontsize=9)

# 右轴：FPS 折线图
ax2 = ax1.twinx()
line = ax2.plot(df["Experiment"], df["Speed (fps)"], color="black", marker='o', linewidth=2, label="Speed (fps)")
ax2.set_ylabel("Speed (fps)", fontsize=12)
ax2.set_ylim(0, max(df["Speed (fps)"])*1.3)
ax2.grid(False)

# ——给折线图添加数据标签——
for i, (x, y) in enumerate(zip(df["Experiment"], df["Speed (fps)"])):
    ax2.text(i, y + 0.3, f'{y:.2f}', color='black', ha='center', va='bottom', fontsize=9)

# 图例合并（左轴+右轴）
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper left')

plt.tight_layout()
plt.show()
