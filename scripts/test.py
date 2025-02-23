import cv2

points = []

def click_event(event, x, y, flags, params):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        points.append((x, y))
        print(f"Point clicked: ({x}, {y})")
        cv2.imshow("Boundary select", frame)
        if len(points) == 4:
            print("Four points selected: ", points)
            cv2.destroyAllWindows()


frame = cv2.imread("/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/input/Athlete/1/near.png")
cv2.imshow("Boundary select", frame)
cv2.setMouseCallback("Boundary select", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Final points:", points)
