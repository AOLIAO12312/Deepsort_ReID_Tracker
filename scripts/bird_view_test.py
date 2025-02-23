import cv2

from src.bird_eye_view import BirdEyeView

bird_view = BirdEyeView()


video_capture = cv2.VideoCapture(
        '/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/input/long_video/Camera2.mp4')
idx = 0
while True:
    ret,frame = video_capture.read()
    if not ret:
        break
    if idx % 1 == 0:
        field_map = bird_view.draw_bird_view(frame)
    idx += 1
    cv2.imshow("Bird Eye View",field_map)
    cv2.waitKey(1)

video_capture.release()
cv2.destroyAllWindows()