import threading
import cv2
import queue
from src.reid_tracker import ReidTracker
from src.utils import draw_reid_tracking_results
reset_queue = queue.Queue()

def wait_for_input():
    while True:
        user_input = input("Please enter the person needs to be reset:")
        reset_queue.put(user_input)


if __name__ == '__main__':
    reid_tracker = ReidTracker(
        "/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/models/yolo/weights/yolo11n.pt",
        "/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/configs/deep_sort.yaml",
        '/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/input/Athlete/', reset_queue, 'cpu')

    video_capture = cv2.VideoCapture(
        '/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/input/long_video/Camera2.mp4')

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = '/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/output/tracked_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

    input_thread = threading.Thread(target=wait_for_input, daemon=True)
    input_thread.start()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        tracking_results = reid_tracker.update(frame.copy())
        draw_reid_tracking_results(tracking_results, frame)
        cv2.imshow("Monitor", frame)
        video_writer.write(frame)
        cv2.waitKey(1)
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"Processing complete, video has been saved as: {output_video_path}")
