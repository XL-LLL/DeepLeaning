# encoding=utf-8
import numpy as np
import cv2
import time
from arm_move import ArmMove


font = cv2.FONT_HERSHEY_SIMPLEX
lower_red = np.array([164, 194, 170])  # 红色阈值下界
higher_red = np.array([180, 217, 190])  # 红色阈值上界
lower_green = np.array([54, 213, 197])  # 黄色阈值下界
higher_green = np.array([66, 233, 216])  # 黄色阈值上界
lower_blue = np.array([95,214,192])
higher_blue = np.array([110,230,210])

armMove = ArmMove()
armMove.reset()

cap = cv2.VideoCapture(0)
color = ""
while (cap.isOpened()):
    ret, frame = cap.read()
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(img_hsv, lower_red, higher_red)  # 可以认为是过滤出红色部分，获得红色的掩膜
    mask_green = cv2.inRange(img_hsv, lower_green, higher_green)  # 获得绿色部分掩膜
    mask_green = cv2.medianBlur(mask_green, 7)  # 中值滤波
    mask_red = cv2.medianBlur(mask_red, 7)  # 中值滤波
    mask_blue = cv2.inRange(img_hsv, lower_blue, higher_blue)  # 获得绿色部分掩膜
    mask_blue = cv2.medianBlur(mask_blue, 7)  # 中值滤波
    cnts1, hierarchy1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓检测 #红色
    cnts2, hierarchy2 = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓检测 #蓝色
    cnts3, hierarchy3 = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 轮廓检测 #绿色

    for cnt in cnts1:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
        cv2.putText(frame, 'red', (x, y - 5), font, 0.7, (0, 0, 255), 2)
        color = "red"
    for cnt in cnts2:

        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
        cv2.putText(frame, 'blue', (x, y - 5), font, 0.7, (0, 0, 255), 2)
        color = "blue"
    for cnt in cnts3:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
        cv2.putText(frame, 'green', (x, y - 5), font, 0.7, (0, 255, 0), 2)
        color = "green"
    cv2.imshow('video', frame)
    if color == "blue":
        color = ""
        armMove.arm_pick(0, 1920, -5760)
        time.sleep(2)
        armMove.arm_release(4800, 1920, -6071)
        time.sleep(1)
        armMove.arm_release(0, 0, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


