#*-coding:utf-8-*-
import cv2 
import sys 
import numpy as np


msg = """


1 : 色彩空间转换
2 : 调整彩色图像的饱和度和亮度
3 : 颜色识别和提取

点击窗口右上角退出

"""

key='1'


print(msg)


def color_space_demo(image):
	cv2.imshow("Source Pic",image)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray", gray)

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	cv2.imshow("hsv", hsv)

	hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	cv2.imshow("hls", hls)

	yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
	cv2.imshow("yuv", yuv)

	ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	cv2.imshow("ycrcb", ycrcb)
	cv2.waitKey(0)

def channels_split_merge(image):
	b, g, r = cv2.split(image)
	cv2.imshow("blue", b)# b通道提取时，对该通道颜色保留，其余通道置为0
	cv2.imshow("green", g)
	cv2.imshow("red", r)

	changed_image = image.copy()
	changed_image[:, :, 2] = 0  # BGR 将r通道颜色全部置为0
	cv2.imshow("changed_image", changed_image)

	merge_image = cv2.merge([b, g, r]) #颜色合并，重回原图
	cv2.imshow("merge_image", merge_image)
	cv2.waitKey(0)


image=cv2.imread('./image/2.jpg',cv2.IMREAD_COLOR)

if key == '1':	
	color_space_demo(image)
elif key == '2' :
	#channels_split_merge(image)
    fImg = image.astype(np.float32)
    fImg = fImg / 255.0
    # 颜色空间转换 BGR转为HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    l = 100
    s = 100
    MAX_VALUE = 100
    # 调节饱和度和亮度的窗口
    cv2.namedWindow("l and s", cv2.WINDOW_AUTOSIZE)
    def nothing(*arg):
        pass
    # 滑动块
    cv2.createTrackbar("l", "l and s", l, MAX_VALUE, nothing)
    cv2.createTrackbar("s", "l and s", s, MAX_VALUE, nothing)
    # 调整饱和度和亮度后的效果
    lsImg = np.zeros(image.shape, np.float32)
    # 调整饱和度和亮度
    while True:
        # 复制
        hlsCopy = np.copy(hlsImg)
        # 得到 l 和 s 的值
        l = cv2.getTrackbarPos('l', 'l and s')
        s = cv2.getTrackbarPos('s', 'l and s')
        # 1.调整亮度（线性变换) , 2.将hlsCopy[:, :, 1]和hlsCopy[:, :, 2]中大于1的全部截取
        hlsCopy[:, :, 1] = (1.0 + l / float(MAX_VALUE)) * hlsCopy[:, :, 1]
        hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
        # 饱和度
        hlsCopy[:, :, 2] = (1.0 + s / float(MAX_VALUE)) * hlsCopy[:, :, 2]
        hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
        # HLS2BGR
        lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
        # 显示调整后的效果
        cv2.imshow("l and s", lsImg)
 
        ch = cv2.waitKey(5)
        # 按 ESC 键退出
        if ch == 27:
            break
        elif ch == ord('s'):
            # 按 s 键保存并退出
            # 保存结果
            lsImg = lsImg * 255
            lsImg = lsImg.astype(np.uint8)
            cv2.imwrite("lsImg.jpg", lsImg)
            break

elif key == '3' :
	ball_color = 'green'

	color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
				  'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
				  'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
				  }
	gs_image = cv2.GaussianBlur(image, (5, 5), 0)                     # 高斯模糊
	hsv = cv2.cvtColor(gs_image, cv2.COLOR_BGR2HSV)                 # 转化成HSV图像
	erode_hsv = cv2.erode(hsv, None, iterations=2)                   # 腐蚀 粗的变细
	inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
	cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	c = max(cnts, key=cv2.contourArea)
	rect = cv2.minAreaRect(c)
	box = cv2.boxPoints(rect)
	cv2.drawContours(image, [np.int0(box)], -1, (0, 255, 255), 2)

	cv2.imshow('image', image)
	cv2.imshow('image2', inRange_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
