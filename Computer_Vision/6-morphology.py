#*-coding:utf-8-*-
import cv2
import sys
import sys, select, os
import numpy as np
import matplotlib.pyplot as plt
import math
msg = """


1 : 腐蚀
2 : 膨胀
3 : 开运算和闭运算、形态学梯度、顶帽变换和底帽变换

点击窗口右上角退出

"""
key=sys.argv[1]


print(msg)


img=cv2.imread('./image/4.png',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('./image/1.jpg',cv2.IMREAD_GRAYSCALE)

if key == '1':
	#腐蚀
	ret,img_thr = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
	'''kernel=cv2.getStructuringElement(shape,ksize,anchor)
        shape:核的形状
                cv2.MORPH_RECT: 矩形
                cv2.MORPH_CROSS: 十字形(以矩形的锚点为中心的十字架)
                cv2.MORPH_ELLIPSE:椭圆(矩形的内切椭圆）
                
        ksize: 核的大小，矩形的宽，高格式为(width,height)
        anchor: 核的锚点，默认值为(-1,-1),即核的中心点'''
	dst = cv2.erode(img_thr,kernel,iterations=1)
	'''dst=cv2.erode(src,kernel,anchor,iterations,borderType,borderValue):
        src: 输入图像对象矩阵,为二值化图像
        kernel:进行腐蚀操作的核，可以通过函数getStructuringElement()获得
        anchor:锚点，默认为(-1,-1)
        iterations:腐蚀操作的次数，默认为1
        borderType: 边界种类，有默认值
        borderValue:边界值，有默认值'''

	cv2.imshow("img",img)
	cv2.imshow("img_thr",img_thr)
	cv2.imshow("dst",dst)

	cv2.waitKey (0)


elif key == '2' :
	#膨胀
	ret,img_thr = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
	dst = cv2.dilate(img_thr,kernel,iterations=1)
	'''dst = cv2.dilate(src,kernel,anchor,iterations,borderType,borderValue)
        src: 输入图像对象矩阵,为二值化图像
        kernel:进行腐蚀操作的核，可以通过函数getStructuringElement()获得
        anchor:锚点，默认为(-1,-1)
        iterations:腐蚀操作的次数，默认为1
        borderType: 边界种类
        borderValue:边界值'''

	cv2.imshow("img",img)
	cv2.imshow("img_thr",img_thr)
	cv2.imshow("dst",dst)
	cv2.waitKey(0)
elif key == '3' :
	#开运算和闭运算、形态学梯度、顶帽变换和底帽变换
	ret,img_thr = cv2.threshold(img2,200,255,cv2.THRESH_BINARY_INV)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
	open = cv2.morphologyEx(img_thr,cv2.MORPH_OPEN,kernel,iterations=1)
	close = cv2.morphologyEx(img_thr,cv2.MORPH_CLOSE,kernel,iterations=1)
	gradient = cv2.morphologyEx(img_thr,cv2.MORPH_GRADIENT,kernel,iterations=1)
	tophat = cv2.morphologyEx(img_thr,cv2.MORPH_TOPHAT,kernel,iterations=1)
	blackhat = cv2.morphologyEx(img_thr,cv2.MORPH_BLACKHAT,kernel,iterations=1)
	'''
	进行开运算，闭运算，顶帽运算，底帽运算，形态学梯度，opencv提供了一个统一的函数cv2.morphologyEx()
	其对应参数如下
	dst = cv2.morphologyEx(src,op,kernel,anchor,iterations,borderType,borderValue)
        src: 输入图像对象矩阵,为二值化图像
        op: 形态学操作类型
            cv2.MORPH_OPEN    开运算
            cv2.MORPH_CLOSE   闭运算
            cv2.MORPH_GRADIENT 形态梯度
            cv2.MORPH_TOPHAT   顶帽运算
            cv2.MORPH_BLACKHAT  底帽运算
            
        kernel:进行腐蚀操作的核，可以通过函数getStructuringElement()获得
        anchor:锚点，默认为(-1,-1)
        iterations:腐蚀操作的次数，默认为1
        borderType: 边界种类
        borderValue:边界值'''

	images=[img_thr,open,close,gradient,tophat,blackhat]
	titles=["img_thr","open","close","gradient","tophat","blackhat"]
	for i in range(6):
	    plt.subplot(2,3,i+1),plt.imshow(images[i],"gray")
	    plt.title(titles[i])
	    plt.xticks([]),    plt.yticks([])
	plt.show()
else:
	print("please input param")
 
cv2.destroyAllWindows()



