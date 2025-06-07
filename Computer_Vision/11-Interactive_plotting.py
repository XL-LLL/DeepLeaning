#*-coding:utf-8-*-
import cv2 
import sys 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from scipy import signal
msg = """


1 : 鼠标绘图  键盘输入 p:点 r：矩形  c：圆  l：线  
2 : 滚动条实现调色板
3 : 滚动条控制阈值处理参数 
4 : 鼠标交互显示ROI


点击窗口右上角退出

"""

key='4'
print(msg)

def getDist_P2P(Point0,PointA):
    distance=math.pow((Point0[0]-PointA[0]),2) + math.pow((Point0[1]-PointA[1]),2)
    distance=math.sqrt(distance)
    return distance

#创建回调函数
def OnMouseAction(event,x,y,flags,param):
    global x1, y1
    print(mode)
    color = np.random.randint(0,high = 256,size = (3,)).tolist()
    if mode == 0 and event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
        cv2.circle(img,(x,y),1,color,2)

    if mode == 1 and event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击1")
        x1, y1 = x, y
    elif mode == 1 and event==cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        print("左鍵拖曳1")
        cv2.rectangle(img,(x1,y1),(x,y),color,-1)
    if mode == 2 and event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击1")
        x1, y1 = x, y
    elif mode == 2 and event==cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        print("左鍵拖曳1")
        r=getDist_P2P((x,y),(x1,y1))
        cv2.circle(img,(int(x*0.5+x1*0.5),int(y*0.5+y1*0.5)),int(r),color,thickness)
    if mode == 3 and event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击1")
        x1, y1 = x, y
    elif mode == 3 and event==cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        print("左鍵拖曳1")
        cv2.line(img,(x1,y1),(x,y),color,3)

def changeColor(x):
	r=cv2.getTrackbarPos('R','image')
	g=cv2.getTrackbarPos('G','image')
	b=cv2.getTrackbarPos('B','image')   # 读取滚动条的值，修改img
	img[:]=[b,g,r]

Type=0  # 阈值处理方式
Value=0 # 使用的阈值
# retval, dst=cv2.threshold(src, thresh, maxval, type)
def onType(a):
    Type= cv2.getTrackbarPos(tType, windowName)
    Value= cv2.getTrackbarPos(tValue, windowName)
    ret, dst = cv2.threshold(o, Value,255, Type) 
    cv2.imshow(windowName,dst)
def onValue(a):
    Type= cv2.getTrackbarPos(tType, windowName)
    Value= cv2.getTrackbarPos(tValue, windowName)
    ret, dst = cv2.threshold(o, Value, 255, Type) 
    cv2.imshow(windowName,dst)




if key == '1':
	thickness=-1
	d=400
	mode=1
	img = cv2.imread('./image/3.jpg')
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',OnMouseAction)
	'''cv2.setMouseCallback(windowName, onMouse [, param])
	参数说明：
	windowName：必需。类似于cv.imshow()函数，opencv具体操作哪个窗口以窗口名作为识别标识，这有点类似窗口句柄的概念。
	onMouse：必需。鼠标回调函数。鼠标回调函数的定义是onMouseAction(event, x, y, flags, param)，
		我们想要做什么鼠标操作，都是在这个函数内实现。
	param：可选。请注意到onMouse里面也有一个param参数，它与是setMouseCallback里的param是同一个，
		更直白一点说，这个param是onMouse和setMouseCallback之间的参数通信接口。
	OnMouseAction()响应函数：
	def OnMouseAction(event,x,y,flags,param):
	Event:  
	EVENT_MOUSEMOVE 0             //滑动  
	EVENT_LBUTTONDOWN 1           //左键点击  
	EVENT_RBUTTONDOWN 2           //右键点击  
	EVENT_MBUTTONDOWN 3           //中键点击  
	EVENT_LBUTTONUP 4             //左键放开  
	EVENT_RBUTTONUP 5             //右键放开  
	EVENT_MBUTTONUP 6             //中键放开  
	EVENT_LBUTTONDBLCLK 7         //左键双击  
	EVENT_RBUTTONDBLCLK 8         //右键双击  
	EVENT_MBUTTONDBLCLK 9         //中键双击  
	int x,int y，代表鼠标位于窗口的（x，y）坐标位置，即Point(x,y);

	int flags，代表鼠标的拖拽事件，以及键盘鼠标联合事件，共有32种事件：

	flags:  
	EVENT_FLAG_LBUTTON 1       //左鍵拖曳  
	EVENT_FLAG_RBUTTON 2       //右鍵拖曳  
	EVENT_FLAG_MBUTTON 4       //中鍵拖曳  
	EVENT_FLAG_CTRLKEY 8       //(8~15)按Ctrl不放事件  
	EVENT_FLAG_SHIFTKEY 16     //(16~31)按Shift不放事件  
	EVENT_FLAG_ALTKEY 32       //(32~39)按Alt不放事件  
	param
	函数指针 标识了所响应的事件函数，相当于自定义了一个OnMouseAction()函数的ID。'''


	while(1):
		cv2.imshow('image',img)
		k=cv2.waitKey(1)
		if k==ord('p'):      # 按p键画点
			mode=0
		elif k==ord('r'):    # 按r键画矩形
			mode=1
		elif k==ord('c'): 	# 按c键画圆
			mode=2
		elif k==ord('l'):	# 按l键画线
			mode=3
		#else:
			#print('请按键选择要画的图像：p  --点  r  --矩形  c --圆  l --线 ')
		elif k==27:
			break 
			print('请按键选择要画的图像：p  --点  r  --矩形  c --圆  l --线 ')
			
elif key == '2' :
	img=np.zeros((100,700,3),np.uint8)
	cv2.namedWindow('image')
	cv2.createTrackbar('R','image',0,255,changeColor)
	cv2.createTrackbar('G','image',0,255,changeColor)
	cv2.createTrackbar('B','image',0,255,changeColor)
	while(1):
		cv2.imshow('image',img)
		k=cv2.waitKey(1)
		if k==27:
			break 
elif key == '3' :
	o = cv2.imread("./image/1.jpg",0)
	windowName = "Demo" #窗体名
	cv2.namedWindow(windowName)
	cv2.imshow(windowName,o)
	tType = "Type" # 用来选取阈值处理方式的滚动条
	tValue = "Value" # 用来选取阈值的滚动条
	cv2.createTrackbar(tType, windowName, 0, 4, onType)
	cv2.createTrackbar(tValue, windowName,0, 255, onValue) 
	print(1)
	if cv2.waitKey(0) == 27:                 # 这里应该是等待按键所以停在了 waitKey() 这。。
	    cv2.destroyAllWindows()

elif key == '4' :
	img = cv2.imread('./image/7.jpg')

	# create a window
	cv2.namedWindow('image')
	cv2.namedWindow('image_roi')

	cv2.imshow('image', img)

	# whether to show crosschair
	showCrosshair = False # 是否显示交叉线
	# if true, then from the center
	# if false, then from the left-top
	fromCenter = False  #是否从中心开始选择
	# then let use to choose the ROI
	rect = cv2.selectROI('image', img, showCrosshair, fromCenter)
	#也可以是 rect = cv2.selectROI('image', img, False, False)#记得改掉上面的语句不要设置为
	# rect = cv2.selectROI('image', img, showCrosshair=False, fromCenter=False)
	# get the ROI
	(x, y, w, h) = rect#(起始x,起始y，终点x，终点y)

	# Crop image
	imCrop = img[int(y) : int(y+h), int(x):int(x+w)]

	# Display cropped image
	cv2.imshow('image_roi', imCrop)

	# write image to local disk
	cv2.imwrite('image_roi.jpg', imCrop)
	cv2.waitKey(0)

cv2.destroyAllWindows()
