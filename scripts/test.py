import cv2

# 定义一个列表来保存鼠标点击的坐标
points = []

# 鼠标回调函数
def click_event(event, x, y, flags, params):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在图像上绘制圆圈标记点击的位置
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        points.append((x, y))
        # 显示点击位置的坐标
        print(f"Point clicked: ({x}, {y})")

        # 更新窗口，展示当前状态
        cv2.imshow("Boundary select", frame)

        # 如果点击了四个点，显示用户已完成
        if len(points) == 4:
            print("Four points selected: ", points)
            cv2.destroyAllWindows()  # 关闭窗口


# 加载图像
frame = cv2.imread("/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/data/input/Athlete/1/near.png")

# 设置窗口标题并绑定鼠标回调函数
cv2.imshow("Boundary select", frame)
cv2.setMouseCallback("Boundary select", click_event)

# 持续显示图像直到四个点被选择
cv2.waitKey(0)
cv2.destroyAllWindows()

# 输出最终选定的四个点
print("Final points:", points)
