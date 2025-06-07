#*-coding:utf-8-*-
import cv2 
import sys 
#导入numpy
import numpy as np
#导入画图工具
import matplotlib.pyplot as plt
#导入支持向量机svm
from sklearn import svm
#导入数据集生成工具
from sklearn.datasets import make_blobs
#导入红酒数据集
from sklearn.datasets import load_wine
msg = """


1 : 核函数为linear的SVM支持向量机
2 : SVM的核函数与参数选择
3 : SVM支持向量机的gamma参数调节

点击窗口右上角退出

"""

key=sys.argv[1]

print(msg)

#定义一个函数用来画图
def make_meshgrid(x,y,h=.02):
	x_min,x_max = x.min() - 1,x.max() + 1
	y_min,y_max = y.min() - 1,y.max() + 1
	xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
	return xx,yy
#定义一个绘制等高线的函数
def plot_contours(ax,clf,xx,yy,**params):
	Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = ax.contourf(xx,yy,Z,**params)
	return out
if key == '1':	
	#先创建50个数据点,让他们分为两类
	X,y = make_blobs(n_samples=50,centers=2,random_state=6)
	#创建一个线性内核的支持向量机模型
	clf = svm.SVC(kernel = 'linear',C=1000)
	clf.fit(X,y)
	#把数据点画出来
	plt.scatter(X[:, 0],X[:, 1],c=y,s=30,cmap=plt.cm.Paired)
	 
	#建立图像坐标
	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	#生成两个等差数列
	xx = np.linspace(xlim[0],xlim[1],30)
	yy = np.linspace(ylim[0],ylim[1],30)
	YY,XX = np.meshgrid(yy,xx)
	xy = np.vstack([XX.ravel(),YY.ravel()]).T
	Z = clf.decision_function(xy).reshape(XX.shape)
	 
	#把分类的决定边界画出来
	ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
	ax.scatter(clf.support_vectors_[:, 0],clf.support_vectors_[:, 1],s=100,linewidth=1,facecolors='none')
	plt.show()

elif key == '2' : 
	#使用酒的数据集
	wine = load_wine()
	#选取数据集的前两个特征
	X = wine.data[:, :2]
	y = wine.target
	 
	C = 1.0 #svm的正则化参数
	models = (svm.SVC(kernel='linear',C=C),svm.LinearSVC(C=C),svm.SVC(kernel='rbf',gamma=0.7,C=C),
			svm.SVC(kernel='poly',degree=3,C=C))
	models = (clf.fit(X,y) for clf in models)
	 
	#设定图题
	titles = ('SVC with linear kernel','LineatSVC (linear kernel)','SVC with RBF kernel',
			'SVC with polynomial (degree 3) kernel')
	 
	#设定一个子图形的个数和排列方式
	flg, sub = plt.subplots(2, 2)
	plt.subplots_adjust(wspace=0.4,hspace=0.4)
	#使用前面定义的函数进行画图
	X0,X1, = X[:, 0],X[:, 1]
	xx,yy = make_meshgrid(X0,X1)
	 
	for clf,title,ax in zip(models,titles,sub.flatten()):
		plot_contours(ax,clf,xx,yy,cmap=plt.cm.plasma,alpha=0.8)
		ax.scatter(X0,X1,c=y,cmap=plt.cm.plasma,s=20,edgecolors='k')
		ax.set_xlim(xx.min(),xx.max())
		ax.set_ylim(yy.min(),yy.max())
		ax.set_xlabel('Feature 0')
		ax.set_ylabel('Featuer 1')
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title(title)
	#将图型显示出来
	plt.show()

elif key == '3' :
	#使用酒的数据集
	wine = load_wine()
	#选取数据集的前两个特征
	X = wine.data[:, :2]
	y = wine.target
	C = 1.0 #svm的正则化参数
	models = (svm.SVC(kernel='rbf',gamma=0.1,C=C),svm.SVC(kernel='rbf',gamma=1,C=C),
			svm.SVC(kernel='rbf',gamma=10,C=C))

	models = (clf.fit(X,y) for clf in models)
	 
	#设定图题
	titles = ('gamma = 0.1','gamma = 1','gamma = 10',)
	 
	#设定一个子图形的个数和排列方式
	flg, sub = plt.subplots(1,3,figsize = (10,3))
	#使用定义好的函数进行画图
	X0,X1, = X[:, 0],X[:, 1]
	xx,yy = make_meshgrid(X0,X1)
	 
	for clf,title,ax in zip(models,titles,sub.flatten()):
		plot_contours(ax,clf,xx,yy,cmap=plt.cm.plasma,alpha=0.8)
		ax.scatter(X0,X1,c=y,cmap=plt.cm.plasma,s=20,edgecolors='k')
		ax.set_xlim(xx.min(),xx.max())
		ax.set_ylim(yy.min(),yy.max())
		ax.set_xlabel('Feature 0')
		ax.set_ylabel('Featuer 1')
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title(title)
	#将图型显示出来
	plt.show()

