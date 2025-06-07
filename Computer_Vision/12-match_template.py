#*-coding:utf-8-*-
import cv2 
import sys 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from scipy import signal
msg = """


1 : 单对象模板匹配
2 : 多对象模板匹配

点击窗口右上角退出

"""

key=sys.argv[1]

print(msg)
if key == '1':	
	img = cv2.imread('./image/7.jpg')
	template = cv2.imread('image_roi.jpg')
	h, w = template.shape[:2]    # rows->h, cols->w
	# 相关系数匹配方法: cv2.TM_CCOEFF
	res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	'''
	    平方差匹配 CV_TM_SQDIFF：用两者的平方差来匹配
	    归一化平方差匹配 CV_TM_SQDIFF_NORMED
	    相关匹配 CV_TM_CCORR：用两者的乘积匹配，数值越大表明匹配程度越好
	    归一化相关匹配 CV_TM_CCORR_NORMED
	    相关系数匹配 CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美匹配，-1表示最差匹配
	    归一化相关系数匹配 CV_TM_CCOEFF_NORMED'''

	left_top = max_loc   # 左上角
	right_bottom = (left_top[0] + w, left_top[1] + h)   # 右下角
	cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置
	cv2.imshow('image',img)
	cv2.imshow('template',template)
	cv2.waitKey(0)
	'''	
	plt.subplot(121), plt.imshow(res, cmap='gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

	plt.subplot(122), plt.imshow(img, cmap='gray')
	plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	plt.show()'''

elif key == '2' :
	img_rgb = cv2.imread('./image/8.jpg')
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	template = cv2.imread('./image/white.jpg',0)
	w, h = template.shape[::-1]
	res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
	threshold = 0.8
	loc = np.where( res >= threshold)
	for pt in zip(*loc[::-1]):
	    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
	cv2.imshow("img",img_rgb)
	cv2.imshow("template",template)
	cv2.waitKey(0)

cv2.destroyAllWindows()
