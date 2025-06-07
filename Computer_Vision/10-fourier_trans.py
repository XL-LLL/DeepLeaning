#*-coding:utf-8-*-
import cv2 
import sys 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from scipy import signal
msg = """


1 : 二维离散的傅里叶（逆）变换 
2 : 傅里叶幅度谱和相位谱
3 : 普残差显著性检测 
4 : 卷积与傅里叶变换


点击窗口右上角退出

"""

key=sys.argv[1]


print(msg)



def fftImage(gray_img, rows, cols):
	rPadded = cv2.getOptimalDFTSize(rows)
	cPadded = cv2.getOptimalDFTSize(cols)
	imgPadded = np.zeros((rPadded, cPadded), np.float32)
	imgPadded[:rows, :cols] = gray_img
	fft_img = cv2.dft(imgPadded, flags=cv2.DFT_COMPLEX_OUTPUT)  #输出为复数，双通道
	return fft_img

#计算幅度谱
def amplitudeSpectrum(fft_img):
	real = np.power(fft_img[:, :, 0], 2.0)
	imaginary = np.power(fft_img[:, :, 1], 2.0)
	amplitude = np.sqrt(real+imaginary)
	return amplitude

#幅度谱的灰度化
def graySpectrum(amplitude):
	amplitude = np.log(amplitude+1)  #增加对比度
	spectrum = cv2.normalize(amplitude,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	spectrum *= 255

	#归一化，幅度谱的灰度化
	# spectrum = amplitude*255/(np.max(amplitude))
	# spectrum = spectrum.astype(np.uint8)

	return spectrum

#计算相位谱并灰度化
def phaseSpectrum(fft_img):
	phase = np.arctan2(fft_img[:,:,1], fft_img[:, :, 0])
	spectrum = phase*180/np.pi  #转换为角度，在[-180,180]之间
	# spectrum = spectrum.astype(np.uint8)
	return spectrum

def stdFftImage(img_gray, rows, cols):
	fimg = np.copy(img_gray)
	fimg = fimg.astype(np.float32)
	# 1.图像矩阵乘以（-1）^(r+c), 中心化
	for r in range(rows):
		for c in range(cols):
			if(r+c)%2:
				fimg[r][c] = -1*img_gray[r][c]
	fft_img = fftImage(fimg, rows, cols)
	amplitude = amplitudeSpectrum(fft_img)
	ampSpectrum = graySpectrum(amplitude)
	return ampSpectrum

img=cv2.imread('./image/1.jpg',cv2.IMREAD_COLOR)

if key == '1':
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	rows, cols = img_gray.shape[:2]

	#快速傅里叶变换，输出复数
	rPadded = cv2.getOptimalDFTSize(rows)
	cPadded = cv2.getOptimalDFTSize(cols)
	imgPadded = np.zeros((rows, cols), np.float32)  # 填充
	imgPadded[:rows, :cols] = img_gray
	fft_img = cv2.dft(imgPadded, flags=cv2.DFT_COMPLEX_OUTPUT)
	print(fft_img)
	'''对于M行N列的图像矩阵，傅里叶变换理论上需要 (m*n)2次运算，非常耗时。
	而当 M=2m 和N=2n 时，傅里叶变换可以通过O(MNlog(M*N)) 次运算就能完成运算，即快速傅里叶变换。
	当图片矩阵的行数和列数都可以分解成 2p*3q*5r时，opencv中的dft()会进行傅里叶变换快速算法，
	所以在计算时一般先对二维矩阵进行扩充补0，已满足规则，
	opencv提供函数getOptimalDFTSize()函数来计算需要补多少行多少列的0，其参数如下：

	retval=cv.getOptimalDFTSize(vecsize)
		vecsize: 整数，图片矩阵的行数或者列数，函数返回一个大于或等于vecsize的最小整数N，且N满足N=2^p*3^q*5^r '''


	#快速傅里叶逆变换，只输出实数部分
	ifft_img = cv2.dft(fft_img, flags=cv2.DFT_REAL_OUTPUT+cv2.DFT_INVERSE+cv2.DFT_SCALE)
	ori_img = np.copy(ifft_img[:rows, :cols])  # 裁剪
	print(img_gray)
	print(ori_img)
	print(np.max(ori_img-img_gray))   #9.1552734e-05，接近于0
	'''
	opencv提供函数dft()可以对图像进行傅里叶变换和傅里叶逆变换，函数参数如下：
	复制代码

	dst =cv.dft(src, flags=0, nonzeroRows=0)
		src: 输入图像矩阵，只支持CV_32F或者CV_64F的单通道或双通道矩阵
			(单通道的代表实数矩阵，双通道代表复数矩阵)
		flags: 转换的标识符，取值包括DFT_COMPLEX_OUTPUT,DFT_REAL_OUTPUT,DFT_INVERSE,
						DFT_SCALE, DFT_ROWS，通常组合使用
			DFT_COMPLEX_OUTPUT: 输出复数形式
			DFT_REAL_OUTPUT: 只输出复数的实部
			DFT_INVERSE:进行傅里叶逆变换
			DFT_SCALE:是否除以M*N （M行N列的图片，共有有M*N个像素点）
			DFT_ROWS:输入矩阵的每一行进行傅里叶变换或者逆变换
			(傅里叶正变换一般取flags=DFT_COMPLEX_OUTPUT，
			 傅里叶逆变换一般取flags= DFT_REAL_OUTPUT+DFT_INVERSE+DFT_SCALE)     
		nonzerosRows: 当设置为非0时，对于傅里叶正变换，表示输入矩阵只有前nonzerosRows行包含非零元素；
				对于傅里叶逆变换，表示输出矩阵只有前nonzerosRows行包含非零元素

	返回值：
		dst: 单通道的实数矩阵或双通道的复数矩阵'''


	new_gray_img = ori_img.astype(np.uint8)

	cv2.imshow("img_gray", img_gray)
	cv2.imshow("new_gray_img", new_gray_img)
	cv2.waitKey(0)
	
elif key == '2' :
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rows, cols = img_gray.shape[:2]
	fft_img = fftImage(img_gray, rows, cols)
	amplitude = amplitudeSpectrum(fft_img)
	ampSpectrum = graySpectrum(amplitude)   #幅度谱灰度化
	phaSpectrum = phaseSpectrum(fft_img)    #相位谱灰度化

	ampSpectrum2 = stdFftImage(img_gray, rows, cols) # 幅度谱灰度化并中心化
	cv2.imshow("img_gray", img_gray)
	cv2.imshow("ampSpectrum", ampSpectrum)
	cv2.imshow("ampSpectrum2", ampSpectrum2)
	cv2.imshow("phaSpectrum", phaSpectrum)
	cv2.waitKey(0)
elif key == '3' :
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rows, cols = img_gray.shape[:2]
	fft_image = fftImage(img_gray, rows, cols)

	# 计算幅度谱及其log值
	amplitude = np.sqrt(np.power(fft_image[:, :, 0], 2.0) + np.power(fft_image[:, :, 1], 2.0)) #计算幅度谱
	ampSpectrum = np.log(amplitude+1)   # 幅度谱的log值

	#计算相位谱, 余弦谱，正弦谱
	phaSpectrum = np.arctan2(fft_image[:,:,1], fft_image[:, :, 0])  # 计算相位谱，结果为弧度
	cosSpectrum = np.cos(phaSpectrum)
	sinSpectrum = np.sin(phaSpectrum)

	#幅度谱灰度级均值平滑
	meanAmpSpectrum= cv2.boxFilter(ampSpectrum, cv2.CV_32FC1, (3, 3))

	#残差
	spectralResidual = ampSpectrum - meanAmpSpectrum
	expSR = np.exp(spectralResidual)

	#实部，虚部，复数矩阵
	real = expSR*cosSpectrum
	imaginary = expSR*sinSpectrum
	new_matrix = np.zeros((real.shape[0], real.shape[1], 2), np.float32)
	new_matrix[:, :, 0] = real
	new_matrix[:, :, 1] = imaginary
	ifft_matrix = cv2.dft(new_matrix, flags=cv2.DFT_COMPLEX_OUTPUT+cv2.DFT_INVERSE)

	#显著性
	# saliencymap = np.sqrt(np.power(ifft_matrix[:, :, 0], 2) + np.power(ifft_matrix[:, :, 1], 2))
	saliencymap = np.power(ifft_matrix[:, :, 0], 2) + np.power(ifft_matrix[:, :, 1], 2)
	saliencymap = cv2.GaussianBlur(saliencymap, (11, 11), 2.5)
	saliencymap = saliencymap/np.max(saliencymap)
	#伽马变换，增加对比度
	saliencymap = np.power(saliencymap, 0.5)
	saliencymap = np.round(saliencymap*255)
	saliencymap = saliencymap.astype(np.uint8)

	cv2.imshow("img_gray", img_gray)
	cv2.imshow("saliencymap", saliencymap)
	cv2.waitKey(0)
elif key == '4' :
	I=np.array([[34,56,1,0,255,230,45,12],[0,201,101,125,52,12,124,12],
				[3,41,42,40,12,90,123,45],[5,245,98,32,34,234,90,123],
				[12,12,10,41,56,89,189,5],[112,87,12,45,78,45,10,1],
				[42,123,234,12,12,21,56,43],[1,2,45,123,10,44,123,90]],np.float64)
	print(I.shape)
	#卷积核
	kernel=np.array([[1,0,-1],[1,0,1],[1,0,-1]],np.float64)
	#I与kernel进行full卷积
	confull=signal.convolve2d(I,kernel,mode='full',boundary='fill',fillvalue=0)
	#I的傅里叶变换
	FT_I=np.zeros((I.shape[0],I.shape[1],2),np.float64)
	cv2.dft(I,FT_I,cv2.DFT_COMPLEX_OUTPUT)
	#kernel的傅里叶变换
	FT_kernel=np.zeros((kernel.shape[0],kernel.shape[1],2),np.float64)
	cv2.dft(kernel,FT_kernel,cv2.DFT_COMPLEX_OUTPUT)
	#傅里叶变换
	fft2=np.zeros((confull.shape[0],confull.shape[1]),np.float64)
	#对I的右侧和下侧补0
	I_Padded=np.zeros((I.shape[0]+kernel.shape[0]-1,I.shape[1]+kernel.shape[1]-1),np.float64)
	I_Padded[:I.shape[0],:I.shape[1]]=I
	FT_I_Padded=np.zeros((I_Padded.shape[0],I_Padded.shape[1],2),np.float64)
	cv2.dft(I_Padded,FT_I_Padded,cv2.DFT_COMPLEX_OUTPUT)
	#对kernel的右侧和下侧补0
	kernel_Padded=np.zeros((I.shape[0]+kernel.shape[0]-1,I.shape[1]+kernel.shape[1]-1),np.float64)
	kernel_Padded[:kernel.shape[0],:kernel.shape[1]]=kernel
	FT_kernel_Padded=np.zeros((kernel_Padded.shape[0],kernel_Padded.shape[1],2),np.float64)
	cv2.dft(kernel_Padded,FT_kernel_Padded,cv2.DFT_COMPLEX_OUTPUT)
	#两个傅里叶变换的对应位置相乘
	FT_Ikernel=cv2.mulSpectrums(FT_I_Padded,FT_kernel_Padded,cv2.DFT_ROWS)
	#利用傅里叶变换求fu11卷积
	ifft2=np.zeros(FT_Ikernel.shape[:2],np.float64)
	cv2.dft(FT_Ikernel,ifft2,cv2.DFT_REAL_OUTPUT+cv2.DFT_INVERSE+cv2.DFT_SCALE)
	l=np.min(ifft2-confull)
	print(int(l))
	#fu1l卷积结果的傅里叶变换与两个傅里叶变换的点相同
	FT_confull=np.zeros((confull.shape[0],confull.shape[1],2),np.float64)
	cv2.dft(confull,FT_confull,cv2.DFT_COMPLEX_OUTPUT)
	print(int(np.min(FT_confull-FT_Ikernel)))
cv2.destroyAllWindows()
