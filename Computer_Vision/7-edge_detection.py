#*-coding:utf-8-*-
import cv2
import sys
import sys, select, os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
msg = """


1 : Roberts算子
2 : Prewitt算子
3 : sobel算子
4 ：Scharr算子
5 ：Krisch算子和Robinson算子
6 ：canny边缘检测
7 : Laplacian算子
8 : 高斯拉普拉斯(LoG)边缘检测
9 : 高斯差分(DoG)边缘检测
10: Marri-Hildreth边缘检测

点击窗口右上角退出

"""
key='8'


print(msg)


img=cv2.imread('./image/5.jpg',0)
img2=cv2.imread('./image/5.jpg',cv2.IMREAD_COLOR)


def robert(img,boundary="symm",fillvalue=0):
	H1,W1 = img.shape[:2]
	r1 = np.array([[1,0],[0,-1]],np.float32)
	r2 = np.array([[0,1],[-1,0]],np.float32)
	H2,W2= 2,2
	#锚点位置
	kr1,kc1=0,0
	con_r1 = signal.convolve2d(img,r1,mode="full",boundary=boundary,fillvalue=fillvalue)
	#截取出same卷积
	con_r1 = con_r1[H2-kr1-1:H1+H2-kr1-1,W2-kc1-1:W1+W2-kc1-1]
	
	kr2,kc2=0,1
	con_r2 = signal.convolve2d(img,r2,mode="full",boundary=boundary,fillvalue=fillvalue)
	con_r2 = con_r2[H2-kr2-1:H1+H2-kr2-1,W2-kc2-1:W1+W2-kc2-1]
	return (con_r1,con_r2)

def prewitt(img,boundary="symm",fillvalue=0):
	H1,W1 = img.shape[:2]
	rx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]],np.float32)
	ry = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],np.float32)
	#也可以分别进行垂直均值平滑卷积，然后水平差分卷积，来加速运算
	con_x = signal.convolve2d(img,rx,mode="same",boundary=boundary,fillvalue=fillvalue)

	con_y = signal.convolve2d(img,ry,mode="same",boundary=boundary,fillvalue=fillvalue)
	
	return (con_x,con_y)


def createLoGKernel(sigma, size):
	H, W = size
	r, c = np.mgrid[0:H:1.0, 0:W:1.0]
	r -= (H-1)/2
	c -= (W-1)/2
	sigma2 = np.power(sigma, 2.0)
	norm2 = np.power(r, 2.0) + np.power(c, 2.0)
	LoGKernel = (norm2/sigma2 -2)*np.exp(-norm2/(2*sigma2))  # 省略掉了常数系数 1\2πσ4

	print(LoGKernel)
	return LoGKernel

def LoG(image, sigma, size, _boundary='symm'):
	LoGKernel = createLoGKernel(sigma, size)
	edge = signal.convolve2d(image, LoGKernel, 'same', boundary=_boundary)
	return edge




# 二维高斯卷积核拆分为水平核垂直一维卷积核，分别进行卷积
def gaussConv(image, size, sigma):
	H, W = size
	# 先水平一维高斯核卷积
	xr, xc = np.mgrid[0:1, 0:W]
	xc = xc.astype(np.float32)
	xc -= (W-1.0)/2.0
	xk = np.exp(-np.power(xc, 2.0)/(2*sigma*sigma))
	image_xk = signal.convolve2d(image, xk, 'same', 'symm')

	# 垂直一维高斯核卷积
	yr, yc = np.mgrid[0:H, 0:1]
	yr = yr.astype(np.float32)
	yr -= (H-1.0)/2.0
	yk = np.exp(-np.power(yr, 2.0)/(2*sigma*sigma))
	image_yk = signal.convolve2d(image_xk, yk, 'same','symm')
	image_conv = image_yk/(2*np.pi*np.power(sigma, 2.0))

	return image_conv

#直接采用二维高斯卷积核，进行卷积
def gaussConv2(image, size, sigma):
	H, W = size
	r, c = np.mgrid[0:H:1.0, 0:W:1.0]
	c -= (W - 1.0) / 2.0
	r -= (H - 1.0) / 2.0
	sigma2 = np.power(sigma, 2.0)
	norm2 = np.power(r, 2.0) + np.power(c, 2.0)
	LoGKernel = (1 / (2*np.pi*sigma2)) * np.exp(-norm2 / (2 * sigma2))
	image_conv = signal.convolve2d(image, LoGKernel, 'same','symm')

	return image_conv

def DoG(image, size, sigma, k=1.1):
	Is = gaussConv(image, size, sigma)
	Isk = gaussConv(image, size, sigma*k)

	# Is = gaussConv2(image, size, sigma)
	# Isk = gaussConv2(image, size, sigma * k)

	doG = Isk - Is
	doG /= (np.power(sigma, 2.0)*(k-1))
	return doG

def zero_cross_default(doG):
	zero_cross = np.zeros(doG.shape, np.uint8);
	rows, cols = doG.shape
	for r in range(1, rows-1):
		for c in range(1, cols-1):
			if doG[r][c-1]*doG[r][c+1] < 0:
				zero_cross[r][c]=255
				continue
			if doG[r-1][c] * doG[r+1][c] <0:
				zero_cross[r][c] = 255
				continue
			if doG[r-1][c-1] * doG[r+1][c+1] <0:
				zero_cross[r][c] = 255
				continue
			if doG[r-1][c+1] * doG[r+1][c-1] <0:
				zero_cross[r][c] = 255
				continue
	return zero_cross

def Marr_Hildreth(image, size, sigma, k=1.1):
	doG = DoG(image, size, sigma, k)
	zero_cross = zero_cross_default(doG)

	return zero_cross


if key == '1':
	con_r1,con_r2 = robert(img)
	con_r1 =np.abs(con_r1) 
	edge_135 = con_r1.astype(np.uint8)
	con_r2 =np.abs(con_r2) 
	edge_45 = con_r2.astype(np.uint8)
	edge = np.sqrt(np.power(con_r1,2.0)+np.power(con_r2,2.0))
	edge = np.round(edge)
	edge[edge>255]=255
	edge = edge.astype(np.uint8)
	cv2.imshow("img",img)
	cv2.imshow("edge_135",edge_135)
	cv2.imshow("edge_45 ",edge_45 )
	cv2.imshow("edge",edge)
	cv2.waitKey(0)


elif key == '2' :
	con_x,con_y = prewitt(img)
	con_x =np.abs(con_x)
	edge_x = con_x.copy()
	edge_x[edge_x>255]=255
	edge_x = con_x.astype(np.uint8)
	con_y =np.abs(con_y) 
	edge_y = con_y.copy()
	edge_y[edge_y>255]=255
	edge_y = con_y.astype(np.uint8)
	#采用插值法方式，将x和y卷积结果合并
	edge = 0.5*con_x+0.5*con_y
	edge[edge>255]=255
	edge = edge.astype(np.uint8)
	cv2.imshow("img",img)
	cv2.imshow("edge_x",edge_x)
	cv2.imshow("edge_y ",edge_y )
	cv2.imshow("edge",edge)
	cv2.waitKey(0)
elif key == '3' :
	#注意此处的ddepth不要设为-1，要设为cv2.CV_32F或cv2.CV_64F，否则会丢失太多信息
	sobel_edge_x = cv2.Sobel(img2,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=5)
	sobel_edge_x = np.abs(sobel_edge_x)
	sobel_edge_x = sobel_edge_x/np.max(sobel_edge_x)
	sobel_edge_x = sobel_edge_x*255  #进行归一化处理
	sobel_edge_x = sobel_edge_x.astype(np.uint8)

	sobel_edge_y = cv2.Sobel(img2,ddepth=cv2.CV_32F,dx=0,dy=1,ksize=5)
	sobel_edge_y = np.abs(sobel_edge_y)
	sobel_edge_y = sobel_edge_y/np.max(sobel_edge_y)
	sobel_edge_y = sobel_edge_y*255
	sobel_edge_y = sobel_edge_y.astype(np.uint8)

	sobel_edge1 = cv2.addWeighted(sobel_edge_x,0.5,sobel_edge_y,0.5,0)

	sobel_edge = cv2.Sobel(img2,ddepth=cv2.CV_32F,dx=1,dy=1,ksize=5)
	sobel_edge = np.abs(sobel_edge)
	sobel_edge = sobel_edge/np.max(sobel_edge)
	sobel_edge = sobel_edge*255
	sobel_edge = sobel_edge.astype(np.uint8)
	'''
	dst = cv2.Sobel(src,ddepth,dx,dy,ksize,scale,delta,borderType)
		src: 输入图像对象矩阵,单通道或多通道
		ddepth:输出图片的数据深度,注意此处最好设置为cv2.CV_32F或cv2.CV_64F
		dx:dx不为0时，img与差分方向为水平方向的Sobel卷积核卷积
		dy: dx=0,dy!=0时，img与差分方向为垂直方向的Sobel卷积核卷积
		dx=1,dy=0: 与差分方向为水平方向的Sobel卷积核卷积
			 dx=0,dy=1: 与差分方向为垂直方向的Sobel卷积核卷积
			 dx=1,dy=1: 分别与垂直和水分方向Sobel卷积核卷积　　　　　　　　
		ksize: sobel核的尺寸，值为1,3,5,7；ksize为1时表示没有平滑算子，只有差分算子
		scale: 放大比例系数
		delta: 平移系数'''


	cv2.imshow("img",img2)
	cv2.imshow("sobel_edge_x",sobel_edge_x)
	cv2.imshow("sobel_edge_y ",sobel_edge_y )
	cv2.imshow("sobel_edge",sobel_edge)
	cv2.imshow("sobel_edge1",sobel_edge1)
	cv2.waitKey(0)
elif key == '4' :
	#注意此处的ddepth不要设为-1，要设为cv2.CV_32F或cv2.CV_64F，否则会丢失太多信息
	scharr_edge_x = cv2.Scharr(img2,ddepth=cv2.CV_32F,dx=1,dy=0) 
	scharr_edge_x = cv2.convertScaleAbs(scharr_edge_x) 
	#convertScaleAbs等同于下面几句：
	# scharr_edge_x = np.abs(scharr_edge_x)
	# scharr_edge_x = scharr_edge_x/np.max(scharr_edge_x)
	# scharr_edge_x = scharr_edge_x*255  #进行归一化处理
	# scharr_edge_x = scharr_edge_x.astype(np.uint8)

	scharr_edge_y = cv2.Scharr(img2,ddepth=cv2.CV_32F,dx=0,dy=1)
	scharr_edge_y = cv2.convertScaleAbs(scharr_edge_y)

	scharr_edge=cv2.addWeighted(scharr_edge_x,0.5,scharr_edge_y,0.5,0) #两者等权叠加 
	'''
　　dst= cv2.Scharr(src,ddepth,dx,dy,scale,delta,borderType)
	src: 输入图像对象矩阵,单通道或多通道
	ddepth:输出图片的数据深度,注意此处最好设置为cv2.CV_32F或cv2.CV_64F
	dx:dx不为0时，img与差分方向为水平方向的Sobel卷积核卷积
	dy: dx=0,dy!=0时，img与差分方向为垂直方向的Sobel卷积核卷积
	
		dx=1,dy=0: 与差分方向为水平方向的Sobel卷积核卷积
		dx=0,dy=1: 与差分方向为垂直方向的Sobel卷积核卷积
	　　　（注意必须满足： dx >= 0 && dy >= 0 && dx+dy == 1）
	
	scale: 放大比例系数
	delta: 平移系数
	borderType:边界填充类型'''

	cv2.imshow("img",img2)
	cv2.imshow("scharr_edge_x",scharr_edge_x)
	cv2.imshow("scharr_edge_y ",scharr_edge_y )
	cv2.imshow("scharr_edge",scharr_edge)
	cv2.waitKey(0)
elif key == '5' :
	#计算Kirsch 边沿检测算子

	#定义Kirsch 卷积模板
	m1 = np.array([[5, 5, 5],[-3,0,-3],[-3,-3,-3]])
	m2 = np.array([[-3, 5,5],[-3,0,5],[-3,-3,-3]])
	m3 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
	m4 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
	m5 = np.array([[-3, -3, -3],[-3,0,-3],[5,5,5]])
	m6 = np.array([[-3, -3, -3],[5,0,-3],[5,5,-3]])
	m7 = np.array([[5, -3, -3],[5,0,-3],[5,-3,-3]])
	m8 = np.array([[5, 5, -3],[5,0,-3],[-3,-3,-3]])

	#周围填充一圈
	#卷积时，必须在原图周围填充一个像素
	img = cv2.copyMakeBorder(img,1,1,1,1,borderType=cv2.BORDER_REPLICATE)

	temp = list(range(8))

	img1 = np.zeros(img.shape) #复制空间  此处必须的重新复制一块和原图像矩阵一样大小的矩阵，以保存计算后的结果

	for i in range(1,img.shape[0]-1):
		for j in range(1,img.shape[1]-1):
			temp[0] = np.abs( ( np.dot( np.array([1,1,1]) , ( m1*img[i-1:i+2,j-1:j+2]) ) ).dot(np.array([[1],[1],[1]])) )
				#利用矩阵的二次型表达，可以计算出矩阵的各个元素之和
			temp[1] = np.abs(
				(np.dot(np.array([1, 1, 1]), (m2 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )
			temp[2] = np.abs( ( np.dot( np.array([1,1,1]) , ( m1*img[i-1:i+2,j-1:j+2]) ) ).dot(np.array([[1],[1],[1]])) )
			temp[3] = np.abs(
				(np.dot(np.array([1, 1, 1]), (m3 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )
			temp[4] = np.abs(
				(np.dot(np.array([1, 1, 1]), (m4 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )
			temp[5] = np.abs(
				(np.dot(np.array([1, 1, 1]), (m5 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )
			temp[6] = np.abs(
				(np.dot(np.array([1, 1, 1]), (m6 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )
			temp[7] = np.abs(
				(np.dot(np.array([1, 1, 1]), (m7 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1],[1],[1]])) )
			img1[i,j] = np.max(temp)
			if img1[i, j] > 255:  #此处的阈值一般写255，根据实际情况选择0~255之间的值
				img1[i, j] = 255
			else:
				img1[i, j] = 0

	cv2.imshow("Krisch",img1)
	print(img.shape)
	cv2.waitKey(0)
elif key == '6' :
	canny_edge1 = cv2.Canny(img, threshold1=60, threshold2=180)
	canny_edge2 = cv2.Canny(img, threshold1=180, threshold2=230)
	canny_edge3 = cv2.Canny(img, threshold1=180, threshold2=230, apertureSize=5, L2gradient=True)
	'''
	edges=cv.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
	image:输入图像对象矩阵,单通道或多通道
	threshold1: 代表双阈值中的低阈值
	threshold2: 代表双阈值中的高阈值
	apertureSize: spbel核的窗口大小，默认为3*3
	L2gradient: 代表计算边缘梯度大小时使用的方式，True代表使用平方和开方的方式，False代表采用绝对值和的方式，默认为False  '''

	cv2.imshow("img", img)
	cv2.imshow("canny_edge1", canny_edge1)
	cv2.imshow("canny_edge2", canny_edge2)
	cv2.imshow("canny_edge3", canny_edge3)
	cv2.waitKey(0)
elif key == '7' :

	dst_img = cv2.Laplacian(img, cv2.CV_32F)
	laplacian_edge = cv2.convertScaleAbs(dst_img)  #取绝对值后，进行归一化
	'''
	dst = cv2.Laplacian(src, ddepth, ksize, scale, delta, borderType)
	src: 输入图像对象矩阵,单通道或多通道
	ddepth:输出图片的数据深度,注意此处最好设置为cv.CV_32F或cv.CV_64F
	ksize: Laplacian核的尺寸，默认为1，采用上面3*3的卷积核
	scale: 放大比例系数
	delta: 平移系数
	borderType: 边界填充类型'''
	cv2.imshow("img", img)
	cv2.imshow("laplacian_edge", laplacian_edge)
	cv2.waitKey(0)
elif key == '8' :
	LoG_edge = LoG(img, 1, (11, 11))
	LoG_edge[LoG_edge>255] = 255
	# LoG_edge[LoG_edge>255] = 0
	LoG_edge[LoG_edge<0] = 0
	LoG_edge = LoG_edge.astype(np.uint8)

	LoG_edge1 = LoG(img, 1, (37, 37))
	LoG_edge1[LoG_edge1 > 255] = 255
	LoG_edge1[LoG_edge1 < 0] = 0
	LoG_edge1 = LoG_edge1.astype(np.uint8)

	LoG_edge2 = LoG(img, 2, (11, 11))
	LoG_edge2[LoG_edge2 > 255] = 255
	LoG_edge2[LoG_edge2 < 0] = 0
	LoG_edge2 = LoG_edge2.astype(np.uint8)

	cv2.imshow("img", img)
	cv2.imshow("LoG_edge", LoG_edge)
	cv2.imshow("LoG_edge1", LoG_edge1)
	cv2.imshow("LoG_edge2", LoG_edge2)
	cv2.waitKey(0)
elif key == '9' :
	sigma = 1
	k = 1.1
	size = (7, 7)
	DoG_edge = DoG(img, size, sigma, k)
	DoG_edge[DoG_edge>255] = 255
	DoG_edge[DoG_edge<0] = 0
	DoG_edge = DoG_edge / np.max(DoG_edge)
	DoG_edge = DoG_edge * 255
	DoG_edge = DoG_edge.astype(np.uint8)

	cv2.imshow("img", img)
	cv2.imshow("DoG_edge", DoG_edge)
	cv2.waitKey(0)
elif key == '10' :
	k = 1.1
	marri_edge = Marr_Hildreth(img, (11, 11), 1, k)
	marri_edge2 = Marr_Hildreth(img, (11, 11), 2, k)
	marri_edge3 = Marr_Hildreth(img, (7, 7), 1, k)

	cv2.imshow("img", img)
	cv2.imshow("marri_edge", marri_edge)
	cv2.imshow("marri_edge2", marri_edge2)
	cv2.imshow("marri_edge3", marri_edge3)
	cv2.waitKey(0)
else:
	print("please input param")
 
cv2.destroyAllWindows()



