#*-coding:utf-8-*-
import cv2
import sys
import sys, select, os
import numpy as np
import matplotlib.pyplot as plt
import math
msg = """


1 : 全局阈值分割
2 : 直方图技术法
3 : 熵算法
4 ：Otsu阈值处理
5 ：自适应阈值
6 ：二值图的逻辑运算

点击窗口右上角退出

"""
key=sys.argv[1]


print(msg)


img=cv2.imread('./image/2.jpg',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('./image/4.png',cv2.IMREAD_GRAYSCALE)

def caleGrayHist(image):
    #灰度图像的高、宽
    rows, cols = image.shape
    #存储灰度直方图
    grayHist = np.zeros([256], np.uint64) #图像的灰度级范围是0~255      
    for r in range(rows):
        
        for c in range(cols):
            
            grayHist[image[r][c]] += 1
            
    return grayHist
 
#直方图技术法
def threshTwoPeaks(image):
    
    #计算灰度直方图
    histogram = caleGrayHist(image)
    
    #找到灰度直方图的最大峰值对应得灰度值
    maxLoc = np.where(histogram==np.max(histogram))
    firstPeak = maxLoc[0][0] #取第一个最大的灰度值
    
    #寻找灰度直方图的第二个峰值对应得灰度值
    measureDists = np.zeros([256], np.float32)
    for k in range(256):
        measureDists[k] = pow(k-firstPeak,2)*histogram[k]
    maxLoc2 = np.where(measureDists==np.max(measureDists))
    secondPeak = maxLoc2[0][0]
    
    #找到两个峰值之间的最小值对应的灰度值，作为阈值
    thresh = 0
    if firstPeak > secondPeak: #第一个峰值在第二个峰值右侧
        temp = histogram[int(secondPeak):int(firstPeak)]
        minLoc = np.where(temp==np.min(temp))
        thresh = secondPeak + minLoc[0][0] + 1 #有多个波谷取左侧的波谷
    else:
        temp = histogram[int(firstPeak):int(secondPeak)]
        minLoc = np.where(temp==np.min(temp))
        thresh = firstPeak + minLoc[0][0] + 1
        
    #找到阈值后进行阈值处理，得到二值图
    threshImage_out = image.copy()
    threshImage_out[threshImage_out > thresh] = 255
    threshImage_out[threshImage_out <= thresh] = 0
    
    return (thresh, threshImage_out)

#熵算法
def threshEntropy(image):
    rows, cols = image.shape
    #求灰度直方图
    grayHist = caleGrayHist(image)
    #归一化灰度直方图，即概率直方图
    normGrayHist = grayHist/float(rows*cols)
    
    #第一步：计算累加直方图，也成为零阶累矩阵
    zeroCumuMoment = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0 :
            zeroCumuMoment[k] = normGrayHist[k]
        else:
            zeroCumuMoment[k] = zeroCumuMoment[k-1] + normGrayHist[k]
    #第二步：计算各个灰度级的熵
    entropy = np.zeros([256], np.float32)
    for k in range(256):
        if k == 0 :
            if normGrayHist[k] == 0 :     
                entropy[k] = 0
            else:
                entropy[k] = - normGrayHist[k] * math.log10(normGrayHist[k])
        else:
            if normGrayHist[k] == 0 :
                entropy[k] = entropy[k-1]
            else:
                entropy[k] = entropy[k-1] - normGrayHist[k] * math.log10(normGrayHist[k])
    #第三步：找阈值
    fT = np.zeros([256], np.float32)
    ft1, ft2 = 0.0, 0.0
    totalEntropy = entropy[255]
    for k in range(255):
        #找最大值
        maxFront = np.max(normGrayHist[0:k+1])
        maxBack = np.max(normGrayHist[k+1:256])
        if maxFront==0 or zeroCumuMoment[k]==0 or maxFront==1 or zeroCumuMoment[k]==1 or totalEntropy==0 :
            ft1 = 0
        else:
            ft1 = entropy[k]/totalEntropy*(math.log10(zeroCumuMoment[k])/math.log10(maxFront))
        if maxBack==0 or 1-zeroCumuMoment[k]==0 or maxBack==1 or 1-zeroCumuMoment[k]==1 :
            ft2 = 0
        else:
            if totalEntropy == 0 :
                ft2 = (math.log10(1-zeroCumuMoment[k])/math.log10(maxBack))
            else:
                ft2 = (1-entropy[k]/totalEntropy)*(math.log10(1-zeroCumuMoment[k])/math.log10(maxBack))
        fT[k] = ft1 + ft2
    
    #找到最大值索引
    threshLoc = np.where(fT==np.max(fT))
    thresh = threshLoc[0][0]
    #阈值处理
    threshold = np.copy(image)
    threshold[threshold > thresh] = 255
    threshold[threshold <= thresh] = 0
    
    return (thresh, threshold)

if key == '1':
	#全局阈值分割
	'''
	threshold(src(单通道矩阵，CV_8U&CV_32F), dst, thresh(阈值), maxVal(在图像二值化显示时，一般=为255), type)
	大于阈值得像素=maxVal，小于&等于阈值得像素=0 (type=THRESH_BINARY)
	大于阈值得像素=0，小于&等于阈值得像素=maxVal (type=THRESH_BINARY_INV)	 
	当(type=THRESH_OTSU & type=THRESH_TRIANGLE(3.x新特性))时，会自动计算阈值。	 
	当(type=THRESH_OTSU + THRESH_BINARY)时，即先用THRESH_OTSU自动计算出阈值，然后利用该阈值采用THRESH_BINARY规则(默认)。''' 	
	#手动设置阈值
	the = 150
	maxval = 255
	the, dst = cv2.threshold(img, the, maxval, cv2.THRESH_BINARY_INV)
	print (the)
	print (dst)

	#Otsu阈值处理
	otsuThe = 0
	otsuThe, dst_Otsu = cv2.threshold(img, otsuThe, maxval, cv2.THRESH_OTSU)
	print (otsuThe)
	print (dst_Otsu)
	 
	#TRIANGLE阈值处理
	triThe = 0
	triThe, dst_tri = cv2.threshold(img, triThe, maxval, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY_INV)
	print (triThe)
	print (dst_tri)

	cv2.imshow("image",img)
	cv2.imshow("out1",dst)
	cv2.imshow("out2",dst_Otsu)
	cv2.imshow("out3",dst_tri)
	cv2.waitKey (0)


elif key == '2' :
	#直方图技术法
	the, dst = threshTwoPeaks(img)
	the1 = 0
	maxval = 255 
	the1, dst1 = cv2.threshold(img, the1, maxval, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)
	print('The thresh is :', the)
	print('The thresh1 is :', the1)
	cv2.imshow("image", img)
	cv2.imshow('thresh_out', dst)
	cv2.imshow('thresh_out1', dst1)
	cv2.waitKey (0)
elif key == '3' :
	#熵算法
	the, dst = threshEntropy(img)
	the1 = 0
	maxval = 255 
	the1, dst1 = cv2.threshold(img, the1, maxval, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)
	print('The thresh is :', the)
	print('The thresh1 is :', the1)
	cv2.imshow("image", img)
	cv2.imshow('thresh_out', dst)
	cv2.imshow('thresh_out1', dst1)
	cv2.waitKey (0)
elif key == '4' :
	#Otsu阈值处理
	triThe = 0
	maxval = 255
	triThe, dst_tri = cv2.threshold(img, triThe, maxval, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
	triThe1, dst_tri1 = cv2.threshold(img, triThe, maxval, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
	print (triThe)
	print (triThe1)
	cv2.imshow("image", img)
	cv2.imshow('thresh_out', dst_tri)
	cv2.imshow('thresh_out1', dst_tri1)
	cv2.waitKey (0)
elif key == '5' :
	#自适应阈值
	dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 143, 0.15)
	cv2.imshow("image", img)
	cv2.imshow('thresh_out', dst)
	cv2.waitKey (0)
elif key == '6' :
	#二值图的逻辑运算
	img2 = cv2.resize(img2,(500,500),interpolation=cv2.INTER_AREA)
	#img2=img
	_and = cv2.bitwise_and(img,img2) #与
	_or = cv2.bitwise_or(img,img2) #或
	_not = cv2.bitwise_not(img) #非（取反）
	_xor = cv2.bitwise_xor(img,img2) #异或
	cv2.imshow("image1", img)
	cv2.imshow("image2", img2)
	cv2.imshow('and',_and)
	cv2.imshow('or',_or)
	cv2.imshow('not',_not)
	cv2.imshow('xor',_xor)
	cv2.waitKey (0)

else:
	print("please input param")
 
cv2.destroyAllWindows()



