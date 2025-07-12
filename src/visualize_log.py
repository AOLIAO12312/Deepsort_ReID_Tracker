import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_speed(csv_path, save_dir="output_charts"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Ensure person_id is string
    df['person_id'] = df['person_id'].astype(str)

    # Smooth velocity with rolling average (window=5)
    df["velocity_smooth"] = df.groupby("person_id")["velocity"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    # Plot smoothed speed
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="frame_idx", y="velocity_smooth", hue="person_id", palette="tab10")
    plt.title("Smoothed Player Speed Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Speed (m/s)")
    plt.legend(title="Player ID")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "speed_over_time_smooth.png"))
    plt.close()

def visualize_trajectories(csv_path, save_dir="output_charts"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['person_id'] = df['person_id'].astype(str)

    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")

    # Plot smoothed trajectories using rolling mean
    for pid, group in df.groupby("person_id"):
        group = group.sort_values("frame_idx")
        group["x_smooth"] = group["x"].rolling(window=5, min_periods=1).mean()
        group["y_smooth"] = group["y"].rolling(window=5, min_periods=1).mean()

        plt.plot(group["x_smooth"], group["y_smooth"], label=f"ID {pid}", linewidth=2)

        # Optional: mark start and end points
        plt.scatter(group["x_smooth"].iloc[0], group["y_smooth"].iloc[0], marker='o', color='gray')
        plt.scatter(group["x_smooth"].iloc[-1], group["y_smooth"].iloc[-1], marker='x', color='black')

    plt.title("Smoothed Player Movement Trajectories")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Optional: match top-down field view
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trajectories_smooth.png"))
    plt.close()

if __name__ == "__main__":
    csv_file = "E:\Deepsort_ReID_Tracker\data\output\keyframe\person_states.csv"  # Your CSV file path here
    visualize_speed(csv_file)
    visualize_trajectories(csv_file)
    print("Smoothed visualizations saved to output_charts/")
