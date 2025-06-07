#*-coding:utf-8-*-
import cv2
import sys
import numpy as np
msg = """

1 : 捕获视频
2 : 播放视频
3 : 保存视频

点击窗口右上角退出

"""
key = sys.argv[1]

print(msg)

if key == '1':
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
elif key == '2':
    cap = cv2.VideoCapture('video/slam.mp4')
    while (cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
elif key == '3':
    cap = cv2.VideoCapture(0)
    # 定义编解码器，创建VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video/output.mp4', fourcc, 20.0, (640, 480))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 0)
            out.write(frame)
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

