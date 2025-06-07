#*-coding:utf-8-*-
import cv2
import sys
import sys, select, os
import numpy as np

msg = """


1 : for affine transformation  仿射变换
2 : for projection transformation 投影变换
3 : for polar coordinates transformation 极坐标变换

点击窗口右上角退出

"""
key='1'


print(msg)


img=cv2.imread('./image/1.jpg',cv2.IMREAD_COLOR)
img2=cv2.imread('./image/2.jpg',cv2.IMREAD_COLOR)
image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("image",image)
h,w=image.shape

if key == '1':
	h,w=image.shape
	#仿射变换矩阵,缩小2倍
	A1=np.array([[0.5,0,0],[0,0.5,0]],np.float32)
	d1=cv2.warpAffine(image,A1,(w,h),borderValue=125)#先缩小2倍,再平移
	A2=np.array([[0.5,0,w/4],[0,0.5,h/4]],np.float32)
	d2=cv2.warpAffine(image,A2,(w,h),borderValue=125)#在d2的基础上,绕图像的中心点旋转
	A3=cv2.getRotationMatrix2D((w/2.0,h/2.0),30,1)
	d3=cv2.warpAffine(d2,A3,(w,h),borderValue=125)
	cv2.imshow("image",image)
	cv2.imshow("d1",d1)
	cv2.imshow("d2",d2)
	cv2.imshow("d3",d3)
	cv2.waitKey(0)

elif key == '2' :
	h,w=image.shape
	src=np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]],np.float32)
	dst=np.array([[50,50],[w/3,50],[50,h-1],[w-1,h-1]],np.float32)
	#计算投影变换矩阵
	p=cv2.getPerspectiveTransform(src,dst)
	#利用计算出的投影变换矩阵进行头像的投影变换
	r=cv2.warpPerspective(image,p,(w,h),borderValue=125)
	#显示原图和投影效果
	cv2.imshow("image",image)
	cv2.imshow("warpPerspective",r)
	cv2.waitKey(0)
elif key == '3' :
	#极坐标变换中心
	linearpolar_img=cv2.linearPolar(img2,(250,250),250,cv2.INTER_LINEAR)
	logpolar_img=cv2.logPolar(img2,(250,250),50,cv2.WARP_FILL_OUTLIERS)
	#显示原图和输出图像
	cv2.imshow("image",img2)
	cv2.imshow("linearpolar_img",linearpolar_img)
	cv2.imshow("logpolar_img",logpolar_img)
	cv2.waitKey(0)
else:
	print("please input param")
cv2.destroyAllWindows()



