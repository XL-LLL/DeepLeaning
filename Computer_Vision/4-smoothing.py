#*-coding:utf-8-*-
import cv2
import sys
import sys, select, os
import numpy as np
import matplotlib.pyplot as plt
import math
msg = """


1 : 图片卷积
2 : 高斯平滑
3 : 均值平滑
4 ：中值平滑
5 ：双边滤波
6 ：联合双边滤波
7 : 导向滤波

点击窗口右上角退出

"""
key=sys.argv[1]

#主函数
#if __name__=="__main__":

print(msg)


img=cv2.imread('./image/1.jpg',cv2.IMREAD_COLOR)
img2=cv2.imread('./image/2.jpg',cv2.IMREAD_COLOR)

def getClosenessWeight(sigma_g, H, W):
	# 计算空间距离权重模板
	r, c = np.mgrid[0:H:1, 0:W:1]  # 构造三维表
	r -= int((H-1) / 2)
	c -= int((W-1) / 2)
	closeWeight = np.exp(-0.5*(np.power(r, 2)+np.power(c, 2))/math.pow(sigma_g, 2))
	return closeWeight
 
def jointBLF(I, H, W, sigma_g, sigma_d, borderType=cv2.BORDER_DEFAULT):
 
	# 构建空间距离权重模板
	closenessWeight = getClosenessWeight(sigma_g, H, W)
 
	# 对I进行高斯平滑
	Ig = cv2.GaussianBlur(I, (W, H), sigma_g)
 
	# 模板的中心点位置
	cH = int((H - 1) / 2)
	cW = int((W - 1) / 2)
 
	# 对原图和高斯平滑的结果扩充边界
	Ip = cv2.copyMakeBorder(I, cH, cH, cW, cW, borderType)
	Igp = cv2.copyMakeBorder(Ig, cH, cH, cW, cW, borderType)
 
	# 图像矩阵的行数和列数
	rows, cols = I.shape
	i, j = 0, 0
 
	# 联合双边滤波的结果
	jblf = np.zeros(I.shape, np.float64)
	for r in range(cH, cH+rows, 1):
		for c in range(cW, cW+cols, 1):
			# 当前位置的值
			pixel = Igp[r][c]
 
			# 当前位置的邻域
			rTop, rBottom = r-cH, r+cH
			cLeft, cRight = c-cW, c+cW
 
			# 从 Igp 中截取该邻域，用于构建相似性权重模板
			region = Igp[rTop: rBottom+1, cLeft: cRight+1]
 
			# 通过上述邻域，构建该位置的相似性权重模板
			similarityWeight = np.exp(-0.5*np.power(region-pixel, 2.0)) / math.pow(sigma_d, 2.0)
 
			# 相似性权重模板和空间距离权重模板相乘
			weight = closenessWeight * similarityWeight
 
			# 将权重归一化
			weight = weight / np.sum(weight)
 
			# 权重模板和邻域对应位置相乘并求和
			jblf[i][j] = np.sum(Ip[rTop:rBottom+1, cLeft:cRight+1]*weight)
 
			j+=1
		j = 0
		i += 1
	return jblf
def guideFilter(I, p, winSize, eps):

	mean_I = cv2.blur(I, winSize)      # I的均值平滑
	mean_p = cv2.blur(p, winSize)      # p的均值平滑

	mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
	mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

	var_I = mean_II - mean_I * mean_I  # 方差
	cov_Ip = mean_Ip - mean_I * mean_p # 协方差

	a = cov_Ip / (var_I + eps)         # 相关因子a
	b = mean_p - a * mean_I            # 相关因子b

	mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
	mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑

	q = mean_a * I + mean_b
	return q

if key == '1':
	kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
	#这个是设置的滤波，也就是卷积核
	out = cv2.filter2D(img,-1,kernel) 
	cv2.imshow("image",img)
	cv2.imshow("out",out)
	cv2.waitKey (0)


elif key == '2' :
	out = cv2.GaussianBlur(img, (5, 5), 2)
	cv2.imshow("image",img)
	cv2.imshow("高斯平滑", out)
	cv2.waitKey (0)
elif key == '3' :
	out = cv2.blur(img, (5, 5))
	cv2.imshow("image",img)
	cv2.imshow("均值平滑", out)
	cv2.waitKey (0)
elif key == '4' :
	out = cv2.medianBlur(img,5)
	cv2.imshow("image",img)
	cv2.imshow("中值平滑", out)
	cv2.waitKey (0)
elif key == '5' :
	out = cv2.bilateralFilter(img, 5, 5, 2)
	cv2.imshow("image",img)
	cv2.imshow("双边滤波", out)
	cv2.waitKey (0)
elif key == '6' :
	I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 将8位图转换为浮点型
	fI = I.astype(np.float64)
 
	# 联合双边滤波，返回值的数据类型为浮点型
	jblf = jointBLF(fI, 33, 33, 7, 2)
	jblf = np.round(jblf)
	out = jblf.astype(np.uint8)
	cv2.imshow("image",img)
	cv2.imshow("联合双边滤波", out)
	cv2.waitKey (0)
elif key == '7' :
	eps = 0.01
	winSize = (16,16)
	#image = cv2.resize(img, None,fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
	I = img/255.0        #将图像归一化
	p =I
	guideFilter_img = guideFilter(I, p, winSize, eps)

	# 保存导向滤波结果
	guideFilter_img  = guideFilter_img  * 255
	guideFilter_img [guideFilter_img  > 255] = 255
	guideFilter_img  = np.round(guideFilter_img )
	out  = guideFilter_img.astype(np.uint8)
	#out = cv2.ximgproc.guidedFilter(img2,image,10,800)
	cv2.imshow("image",img)
	cv2.imshow("导向滤波", out )
	cv2.waitKey (0)

else:
	print("please input param")
 
cv2.destroyAllWindows()



