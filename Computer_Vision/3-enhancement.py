#*-coding:utf-8-*-
import cv2
import sys
import sys, select, os
import numpy as np
import matplotlib.pyplot as plt
msg = """


1 : 灰度直方图
2 : 线性变换
3 : 直方图正规化
4 ：伽马变换
5 ：全局直方图均衡化
6 ：限制对比度的自适应直方图均衡化

点击窗口右上角退出

"""
key='2'

#主函数
#if __name__=="__main__":

print(msg)


img=cv2.imread('./image/1.jpg',cv2.IMREAD_COLOR)
img2=cv2.imread('./image/3.jpg',cv2.IMREAD_COLOR)


if key == '1':

	cv2.imshow("image",img)
	#cv2.waitKey(0)
	plt.hist(img.ravel(), 256)
	#.hist()作用是绘制直方图
	#.ravel()作用是将多维数组降为一维数组，格式为：一维数组 = 多维数组.ravel()
	plt.show()

elif key == '2' :
	out = 2.0 * img
	# 进行数据截断，大于255的值截断为255
	out[out > 255] = 255
	# 数据类型转换
	out = np.around(out)
	out = out.astype(np.uint8)
	cv2.imshow("img", img)
	cv2.imshow("out", out)
	#cv2.waitKey(0)
	plt.subplot(1, 2, 1)
	plt.hist(img.ravel(), 256)		
	#lt.show()
	plt.subplot(1, 2, 2)
	plt.hist(out.ravel(), 256)
	plt.show()

elif key == '3' :
	#将图片转换为灰度图像
	image = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	# 计算原图中出现的最小灰度级和最大灰度级
	# 使用函数计算
	Imin, Imax = cv2.minMaxLoc(image)[:2]
	# 使用numpy计算
	# Imax = np.max(img)
	# Imin = np.min(img)
	Omin, Omax = 0, 255
	# 计算a和b的值
	a = float(Omax - Omin) / (Imax - Imin)
	b = Omin - a * Imin
	out = a * image + b
	out = out.astype(np.uint8)
	cv2.imshow("img", image)
	cv2.imshow("out", out)
	#cv2.waitKey(0)
	plt.subplot(1, 2, 1)
	plt.hist(image.ravel(), 256)		
	plt.subplot(1, 2, 2)
	plt.hist(out.ravel(), 256)
	plt.show()
elif key == '4' :
	# 图像归一化
	fi = img / 255.0
	# 伽马变换
	gamma = 0.4
	out = np.power(fi, gamma)
	cv2.imshow("img", img)
	cv2.imshow("out", out)
	plt.subplot(1, 2, 1)
	plt.hist(img.ravel(), 256)		
	plt.subplot(1, 2, 2)
	plt.hist(out.ravel(), 256)
	plt.show()
elif key == '5' :
	 # 如果想要对图片做均衡化，必须将图片转换为灰度图像
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 使用全局直方图均衡化
	dst = cv2.equalizeHist(gray)  # 在说明文档中有相关的注释与例子
	# equalizeHist(src, dst=None)函数只能处理单通道的数据,src为输入图像对象矩阵，必须为单通道的uint8类型的矩阵数据
	# 2dst: 输出图像矩阵(src的shape一样)
	cv2.imshow("global equalizeHist", dst)
	plt.hist(img.ravel(), 256)
	#.hist()作用是绘制直方图
	#.ravel()作用是将多维数组降为一维数组，格式为：一维数组 = 多维数组.ravel()
	plt.show()

elif key == '6' :
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, None, fx=0.5, fy=0.5)
	# 创建CLAHE对象
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	# 限制对比度的自适应阈值均衡化
	dst = clahe.apply(img)

	# 分别显示原图，CLAHE
	cv2.imshow("img", img)
	cv2.imshow("dst", dst)

	plt.subplot(1, 2, 1)
	plt.hist(img.ravel(), 256)
	#.hist()作用是绘制直方图
	#.ravel()作用是将多维数组降为一维数组，格式为：一维数组 = 多维数组.ravel()
	plt.subplot(1, 2, 2)
	plt.hist(dst.ravel(), 256)
	plt.show()

else:
	print("please input param")
for i in range(1,5): 
	cv2.destroyAllWindows()
	cv2.waitKey (1)


