from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip

def concat_videos_center_align(video_path1, video_path2, scale1=1.0, scale2=1.0, output_path="E:\\Deepsort_ReID_Tracker\\data\\output\\center_aligned_output.mp4"):
    # 加载视频并进行缩放
    clip1 = VideoFileClip(video_path1).resize(scale1)
    clip2 = VideoFileClip(video_path2).resize(scale2)

    # 统一时长为最短的那一个
    final_duration = min(clip1.duration, clip2.duration)
    clip1 = clip1.subclip(0, final_duration)
    clip2 = clip2.subclip(0, final_duration)

    # 获取尺寸
    w1, h1 = clip1.size
    w2, h2 = clip2.size
    total_width = w1 + w2
    total_height = max(h1, h2)

    # 创建白色背景
    background = ColorClip(size=(total_width, total_height), color=(255, 255, 255), duration=final_duration)

    # 左边 clip1 顶部对齐，右边 clip2 垂直居中对齐
    clip1 = clip1.set_position((0, 0))
    clip2 = clip2.set_position((w1, (total_height - h2) // 2))  # 垂直居中

    # 合并
    final = CompositeVideoClip([background, clip1, clip2])
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

# 示例调用：clip1 缩放为 100%，clip2 缩放为 70%
concat_videos_center_align("E:\\Deepsort_ReID_Tracker\\data\\output\\ablation_output\\tracked_video_GRID_full_model.mp4", "E:\\Deepsort_ReID_Tracker\\data\\output\\ablation_output\\tracked_video_MIXED_full_model.mp4", scale1=1.0, scale2=0.7, output_path="center_aligned_output.mp4")
