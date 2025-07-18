import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_offense_lines(csv_path, save_dir="output_charts"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['person_id'] = df['person_id'].astype(str)
    df['state'] = df['state'].astype(str)

    player_ids = sorted(df['person_id'].unique())
    player_id_map = {pid: i for i, pid in enumerate(player_ids)}  # map to y position

    plt.figure(figsize=(14, 6))

    for pid in player_ids:
        player_data = df[df["person_id"] == pid]
        attack_frames = player_data[player_data["state"] == "ATTACK"]["frame_idx"].values

        if len(attack_frames) == 0:
            continue

        # group into continuous sequences
        sequences = []
        current_seq = [attack_frames[0]]
        for i in range(1, len(attack_frames)):
            if attack_frames[i] == attack_frames[i-1] + 1:
                current_seq.append(attack_frames[i])
            else:
                sequences.append(current_seq)
                current_seq = [attack_frames[i]]
        sequences.append(current_seq)

        # draw each continuous segment as a line
        for seq in sequences:
            plt.hlines(y=player_id_map[pid], xmin=seq[0], xmax=seq[-1], color='green', linewidth=4)

    plt.yticks(list(player_id_map.values()), list(player_id_map.keys()))
    plt.xlabel("Frame Index")
    plt.ylabel("Player ID")
    plt.title("Offensive State Timeline (Only one player allowed to ATTACK)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "offense_timeline_lines.png"))
    plt.close()
    print("✅ Offensive timeline (line version) saved to output_charts/")

if __name__ == "__main__":
    csv_file = "E:\Deepsort_ReID_Tracker\data\output\keyframe\person_states_long_video.csv"
    visualize_offense_lines(csv_file)
