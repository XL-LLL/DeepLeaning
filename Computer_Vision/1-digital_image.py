#*-coding:utf-8-*-
import cv2 
import sys 
import numpy as np


msg = """


1 : for gray image
2 : for color image

点击窗口右上角退出

"""
key='2'


print(msg)
	
if key == '2':
	image=cv2.imread('./image/1.jpg',cv2.IMREAD_COLOR)

	#得到三个颜色通道
	b=image[:,:,0]
	g=image[:,:,1]
	r=image[:,:,2]
	#显示三个颜色通道
	cv2.imshow("b",b)
	cv2.imshow("g",g)
	cv2.imshow("r",r)
	cv2.imshow("image",image)
elif key == '1' :
	image=cv2.imread('./image/gray.jpg',cv2.IMREAD_GRAYSCALE)
	cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
