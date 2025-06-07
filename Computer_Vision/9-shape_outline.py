#*-coding:utf-8-*-
import cv2 
import sys 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

msg = """


1 : 点集的最小外包 
2 : 霍夫直线检测
3 : 霍夫圆检测
4 : 寻找和绘制轮廓
5 : 轮廓周长和面积  点和轮廓的位置关系
6 : 轮廓的凸包缺陷


点击窗口右上角退出

"""

key=sys.argv[1]


print(msg)

img=cv2.imread('./image/1.jpg',cv2.IMREAD_COLOR)
img4=cv2.imread('./image/4.jpg',cv2.IMREAD_COLOR)
img3=cv2.imread('./image/3.jpg',cv2.IMREAD_COLOR)
img2=cv2.imread('./image/2.jpg',cv2.IMREAD_COLOR)
img_copy1=img3.copy()
img_copy2=img3.copy()
img_copy3=img3.copy()
img_copy4=img3.copy()
def draw_rect(img, points):
	center, size, angle = cv2.minAreaRect(points)   #中心点坐标，尺寸，旋转角度
	print(center, size, angle)
	vertices= cv2.boxPoints((center, size, angle))
	print(vertices)

	for i in range(4):
		point1 = vertices[i, :]
		point2 = vertices[(i+1)%4, :]
		cv2.line(img, tuple(point1), tuple(point2), (0, 0, 255), 2)

	cv2.imshow("最小外包旋转矩形", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if key == '1':
	image = Image.open("./image/3.jpg")
	plt.imshow(image)
	print('请用鼠标点击选择图片上的四个点')
	pos=plt.ginput(4)
	print(pos)
	points = np.array(pos,np.int32)
	#最小外包旋转矩形
	center, size, angle = cv2.minAreaRect(points)   #中心点坐标，尺寸，旋转角度
	print(center,size,angle)
	vertices= cv2.boxPoints((center, size, angle))
	print(vertices)
	for i in range(4):
		point1 = vertices[i, :]
		point2 = vertices[(i+1)%4, :]
		cv2.line(img3, tuple(point1), tuple(point2), (0, 0, 255), 2)

	cv2.imshow("最小外包旋转矩形", img3)
	#最小外包直立矩形
	rect = cv2.boundingRect(points)
	print(rect)
	'''
	rect = cv2.boundingRect(points)
	points: 坐标点array，数据类型为数据类型为int32或者float32
	rect: 矩形的左上角坐标，宽，高 (x, y, w, h)
   '''
	x, y, w, h = rect
	cv2.rectangle(img_copy1,(x, y), (x+w, y+h), (0, 0, 255), 2)
	cv2.imshow("最小外包直立矩形", img_copy1)
	#最小外包圆
	center, radius = cv2.minEnclosingCircle(points)
	print(center, radius)
	'''
	center, radius=cv2.minEnclosingCircle(points)
	points: 坐标点array，数据类型为数据类型为int32或者float32
	center：圆心坐标 
	radius: 圆直径 '''
	cv2.circle(img_copy2, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)  
	# 传入center和radius需要为整形
	cv2.imshow("最小外包圆", img_copy2)
	#最小外包三角形 
	points = points.reshape(4, 1, 2) 
	area, triangle = cv2.minEnclosingTriangle(points)
	print(area, triangle, triangle.shape)

	'''area, triangle=cv.minEnclosingTriangle(points)
	points: 坐标点array，数据类型为数据类型为int32或者float32，注意shape必须为n*1*2
	area: 三角形的顶点
	triangle: 三角形的三个顶点'''

	for i in range(3):
		point1 = triangle[i, 0, :]
		point2 = triangle[(i+1)%3, 0, :]
		print(point1)
		cv2.line(img_copy3, tuple(point1), tuple(point2), (0, 0, 255), 2)

	cv2.imshow("最小外包三角形 ", img_copy3)
	#最小凸包
	plt.close()
	image2 = Image.open("./image/1.jpg")
	plt.imshow(image2)
	print('请用鼠标点击选择图片上的八个点')
	pos=plt.ginput(8)
	print(pos)
	points = np.array(pos,np.int32)

	hull = cv2.convexHull(points)
	n = hull.shape[0]
	for i in range(n):
		point1 = hull[i, 0, :]
		point2 = hull[(i+1)%n, 0, :]
		cv2.line(img, tuple(point1), tuple(point2), (0,0,255), 2)
	cv2.imshow("最小凸包", img)
	cv2.waitKey(0)
	
elif key == '2' :
	img_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
	img_edge = cv2.Canny(img_gray, threshold1=350, threshold2=400, apertureSize=3)
	lines = cv2.HoughLines(img_edge, rho=1, theta=math.pi/180, threshold=250)
	print(lines.shape)
	'''
	lines=cv.HoughLines(image, rho, theta, threshold, srn=0, stn=0, min_theta=0, max_theta=CV_PI)

	image: 单通道的灰度图或二值图
	rho: 距离步长，单位为像素(上述投票器中纵轴)
	theta: 角度步长，单位为弧度 (上述投票器中横轴)
	threshold: 投票器中计数阈值，当投票器中某个点的计数超过阈值，则认为该点对应图像中的一条直线，
		也可以理解为图像空间空中一条直线上的像素点个数阈值(如设为5，则表示这条直线至少包括5个像素点)
	srn:默认为0， 用于在多尺度霍夫变换中作为参数rho的除数，rho=rho/srn
	stn：默认值为0，用于在多尺度霍夫变换中作为参数theta的除数，theta=theta/stn
		(如果srn和stn同时为0，就表示HoughLines函数执行标准霍夫变换，否则就是执行多尺度霍夫变换)
	min_theta: 默认为0，表示直线与坐标轴x轴的最小夹角为0
	max_theta：默认为CV_PI，表示直线与坐标轴x轴的最大夹角为180度   
	lines：返回的直线集合，每一条直线表示为 (ρ,θ) or (ρ,θ,votes)，ρ表示(0,0)像素点到该直线的距离，
		θ表示直线与坐标轴x轴的夹角，votes表示投票器中计数值'''
	for line in lines:
		rho = line[0][0]
		theta = line[0][1]
		a = math.cos(theta)
		b = math.sin(theta)
		x0 = rho*a
		y0 = rho*b  # 原点到直线的垂线，与直线的交点
		x1 = int(x0+1000*(-b))   # 取1000长度，在（x0, y0）上下从直线中各取一点 (由于图像坐标系y轴反向，所以为-b)
		y1 = int(y0+1000*a)
		x2 = int(x0 - 1000 * (-b))
		y2 = int(y0 - 1000 * a)
		cv2.line(img4, (x1, y1), (x2, y2), (0, 0, 255), 2)

	cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	cv2.imshow("img",img4)
	cv2.namedWindow("img_edge", cv2.WINDOW_NORMAL)
	cv2.imshow("img_edge",img_edge)
	cv2.waitKey(0)
elif key == '3' :
	img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 2, 30, param1=400, param2=200, minRadius=20)
	print(circles.shape)
	for circle in circles[0]:
		center_x, center_y, radius = circle
		#cv2.circle(img2, (center_x, center_y), int(radius),(0, 0, 255), 2)

	circles2 = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT_ALT, 2, 30, param1=400, param2=0.95, minRadius=20)
	print(circles2.shape)
	'''
	circles =cv2.HoughCircles(image, method, dp, minDist, param1=100, param2=100, minRadius=0, maxRadius=0)
		image: 单通道灰度图
		method: 霍夫圆检测方法，包括HOUGH_GRADIENT和HOUGH_GRADIENT_ALT
		dp: 图片的分辨率和投票器的分辨率比值，dp=2表示投票器的宽高为图片的一半；
			对于HOUGH_GRADIENT_ALT方法，推荐dp=1.5
		minDist: 圆心之间的最小距离，距离太小时会产生很多相交的圆，距离太大时会漏掉正确的圆
		param1:canny边缘检测双阈值中的高阈值,低阈值默认是其一半
		param2: 对于HOUGH_GRADIENT方法，投票器计数阈值(基于圆心和半径的投票器)；
			对于HOUGH_GRADIENT_ALT方法，表示接近一个圆的程度，param2越趋近1，拟合形状越趋近于圆，
			推荐param2=0.9
		minRadius:需要检测圆的最小半径
		maxRadius:需要检测圆的最大半径，maxRadius<=0时采用图片的最大尺寸(长或宽)
		（HOUGH_GRADIENT_ALT是对HOUGH_GRADIENT的改进，提取圆效果更好）   
	circles: 返回N个圆的信息，存储在1*N*3，每个圆组成为(x, y, radius),其中(x, y)为圆心，radius为半径'''

	for circle in circles2[0]:
		center_x, center_y, radius = circle
		cv2.circle(img2, (center_x, center_y), int(radius),(0, 0, 255), 2)

	cv2.imshow("img", img2)
	cv2.waitKey(0)
elif key == '4' :
	img_cp1 = img2.copy()
	img_cp2 = img2.copy()
	img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	img_gaussian = cv2.GaussianBlur(img_gray, (3, 3), 1)
	edge = cv2.Canny(img_gaussian, 250, 300)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
	edge = cv2.dilate(edge, kernel, iterations=2) #横向的形态学膨胀
	# thre, edge = cv2.threshold(img_gaussian, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

	#寻找轮廓
	contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#绘制检测到的轮廓
	cv2.drawContours(img_cp1, contours, -1, (0,0,255),2)
	'''
	contours, hierarchy =cv2.findContours(image, mode, method, offset=None)
     	    image: 单通道的二值图(若输入灰度图，非零像素点会被设成1，0像素点设成0)
	    
	    mode: 轮廓检索模式，包括RETR_EXTERNAL，RETR_LIST，RETR_CCOMP，RETR_TREE，RETR_FLOODFILL
		RETR_EXTERNAL: 只检索最外层的轮廓 (返回值会设置所有hierarchy[i][2]=hierarchy[i][3]=-1)
		RETR_LIST: 检索所有的轮廓，但不建立轮廓间的层次关系(hierarchy relationship)
		RETR_CCOMP:  检测所有的轮廓，但只建立两个等级关系，外围为顶层，
			若外围内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层，
			只有内围轮廓不再包含子轮廓时，其为内层。
		RETR_TREE:检测所有轮廓，所有轮廓建立一个等级树结构。外层轮廓包含内层轮廓，内层轮廓还可以继续包含内嵌轮廓。
		RETR_FLOODFILL:
		
	    method: 轮廓的近似方法，包括CHAIN_APPROX_NONE，CHAIN_APPROX_SIMPLE，
				CHAIN_APPROX_TC89_L1，CHAIN_APPROX_TC89_KCOS
		CHAIN_APPROX_NONE: 保存物体边界上所有连续的轮廓点到contours中，即点(x1,y1)和点(x2,y2)，
				满足max(abs(x1-x2),abs(y2-y1))==1，则认为其是连续的轮廓点
		CHAIN_APPROX_SIMPLE: 仅保存轮廓的拐点信息到contours，拐点与拐点之间直线段上的信息点不予保留
		CHAIN_APPROX_TC89_L1: 采用Teh-Chin chain近似算法 
		CHAIN_APPROX_TC89_KCOS:采用Teh-Chin chain近似算法
		
	    offset:偏移量，所有的轮廓信息相对于原始图像的偏移量，相当于在每一个检测出的轮廓点上加上该偏移量
			(在图片裁剪时比较有用)

	    
	返回值：
	    contours:返回的轮廓点集合，一个list，每一个元素是一个轮廓，轮廓是一个N*1*2的ndarray
	    hierarchy: 轮廓之间的层次关系，每一个元素对应contours中相应索引轮廓的层次关系，是一个N*4的array，
			hierarchy[i][0]~hierarchy[i][3]分别表示第i个轮廓的后一个轮廓，前一个轮廓，
			第一个内嵌轮廓(子轮廓),父轮廓的索引编号，如果当前轮廓没有对应的后一个轮廓、
			前一个轮廓、内嵌轮廓或父轮廓，则hierarchy[i][0]~hierarchy[i][3]的相应位被设置为默认值-1。

	　　Teh-Chin chain近似算法: C-H Teh and Roland T. Chin. On the detection of dominant points on digital curves. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 11(8):859–872, 1989.


	image=cv2.drawContours(image, contours, contourIdx, color, thickness=None, 
			lineType=None, hierarchy=None, maxLevel=None, offset=None)

	    image: 绘制的轮廓的图像矩阵
	    contours: 所有的轮廓集合（findContours()返回值）
	    contourIdx: 轮廓集合的索引，表示指定一个轮廓进行绘制；若为负数，表示绘制所有轮廓
	    color: 绘制使用的颜色
	    thickness：线的粗细
	    lineType: 线的类型，包括FILLED，LINE_4，LINE_8，LINE_AA
	    hierarchy: 轮廓的层次关系（findContours()返回值）
	    maxLevel: 0表示只有指定的轮廓被绘制，1表示绘制指定的轮廓和其第一层内嵌轮廓，
		2表示绘制指定轮廓和其所有的内嵌轮廓（只有hierarchy部位None时，才起作用）
	    offset: 绘制轮廓时的偏移量'''

	#轮廓拟合
	num = len(contours)
	print(num)
	for i in range(num):
		area = cv2.contourArea(contours[i], oriented=False)
		if 30 < area < 40000:  #限定轮廓的面积
			rect = cv2.boundingRect(contours[i])
			print(rect)
			cv2.rectangle(img_cp2, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0),2)


	# cv2.imshow("img_gray", img_gray)
	cv2.imshow("img", img2)
	cv2.imshow("img_contour", img_cp1)
	cv2.imshow("img_rect", img_cp2)
	cv2.imshow("edge", edge)
	cv2.waitKey(0)
elif key == '5' :
	image = Image.open("./image/3.jpg")
	plt.imshow(image)
	print('请用鼠标点击选择图片上的六个点')
	pos=plt.ginput(6)
	print(pos)
	points = np.array(pos,np.int32)
	length1 = cv2.arcLength(points, False)  #首尾不相连
	length2 = cv2.arcLength(points, True)  #首尾相连
	print("周长",length1,length2)  #324.3223342895508 424.3223342895508

	area1 = cv2.contourArea(points, oriented=True)  #返回点集排列顺序
	area2 = cv2.contourArea(points, oriented=False)
	print("面积",area1, area2)  #-7500.0 7500.0
	'''
	opencv提供函数arcLength()来计算点集所围区域的周长，其参数如下：

	retval=cv2.arcLength(curve, closed)
	    curve: 坐标点集，n*2的array
	    closed: 点集所围区域是否时封闭的

	opencv提供函数contourArea() 来计算点集所围区域的面积，其参数如下：

	retval=cv2.contourArea(contour, oriented=False)
	    contour: 组成轮廓的坐标点集
            oriented: 为True时，返回的面积会带有符号，正数表示轮廓点顺时针排列，负数表示逆时针排列；
	为False时，返回面积的绝对值，不带符号'''

	p1 = (100, 100)
	p1_ret = cv2.pointPolygonTest(points, p1, measureDist=False)
	p1_ret2 = cv2.pointPolygonTest(points, p1, measureDist=True)
	print(p1_ret, p1_ret2) # -1.0 -28.284271247461902

	p2 = (250, 340)
	p2_ret = cv2.pointPolygonTest(points, p2, measureDist=False)
	p2_ret2 = cv2.pointPolygonTest(points, p2, measureDist=True)
	print(p2_ret, p2_ret2)  #1.0 20.0
	'''
	opencv提供函数pointPolygonTest()来计算坐标点和一个轮廓的位置关系，其参数如下：

	retval=cv.pointPolygonTest(contour, pt, measureDist)
	    contour: 组成轮廓的点集
	    pt: 坐标点
	    measureDist: 为False时，返回值为1，-1，0(1表示点在轮廓内，-1表示点在轮廓外面，0在轮廓边缘上)；
			为True时，返回坐标点离轮廓边缘的最小距离(带符号，分别表示轮廓内和轮廓外)'''

	cv2.circle(img3, p1, radius=2, color=(0, 0, 255), thickness=2)
	cv2.circle(img3, p2, radius=2, color=(0, 0, 255), thickness=2)


	rows, cols = points.shape
	for i in range(rows):
	    point1 = tuple(points[i])
	    point2 = tuple(points[(i+1)%rows])
	    cv2.circle(img3, point1, radius=2, color=(55, 55, 55), thickness=2)
	    cv2.line(img3, point1, point2, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_4)

	cv2.imshow("img", img3)
	cv2.waitKey(0)
elif key == '6' :
	img = Image.open("./image/3.jpg")
	plt.imshow(img)
	print('请用鼠标按轮廓边缘顺序点击选择图片上六个点')
	pos=plt.ginput(6)
	print(pos)
	points = np.array(pos,np.int32)
	print(points)
	#points = np.array([[10, 120],[150, 170],[120, 220],[220,220],[200,170],[220,120]], np.int32)
	hull = cv2.convexHull(points,returnPoints=False)  # 返回索引
	defects = cv2.convexityDefects(points, hull)
	print(hull)
	print(defects)
	'''
	convexHull()函数能检测出点集的最小凸包，opencv还提供了函数convexityDefects()来检测凸包的缺陷，
			这里缺陷指凸包的内陷处，
	convexityDefects()函数的参数如下：

	convexityDefects=cv2.convexityDefects(contour, convexhull)
	    contour: 组成轮廓的点集(有序的点集)
	    convexhull: convexHull()的返回值，代表组成凸包的的坐标点集索引

	返回值：    
	    convexityDefects:n*1*4的array，每一个元素代表一个缺陷，
	    缺陷包括四个值：缺陷的起点，终点和最远点的索引，最远点到凸包的距离 
	    (返回的距离值放大了256倍，所以除以256才是实际的距离）
	'''


	rows, cols = points.shape
	for i in range(rows):
	    point1 = tuple(points[i])
	    point2 = tuple(points[(i+1)%rows])
	    cv2.circle(img3, point1, radius=2, color=(55, 55, 55), thickness=2)
	    cv2.line(img3,point1, point2, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_4)
	rows2, _ = hull.shape
	for j in range(rows2):
	    index1 = hull[j][0]
	    index2 = hull[(j+1)%rows2][0]
	    point1 = tuple(points[index1])
	    point2 = tuple(points[index2])
	    cv2.line(img3,point1, point2, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_4)


	cv2.imshow("img", img3)
	cv2.waitKey(0)
cv2.destroyAllWindows()
