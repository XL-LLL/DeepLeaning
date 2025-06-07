#*-coding:utf-8-*-
import matplotlib.pyplot as plt
#from knn import *
#本例中需要用到鸢尾花数据集，它包含在scikit-learn的datasets模块中。可以调用load_iris函数来加载函数
#load_iris返回的iris对象是一个Bunch对象，与字典非常相似，里面包含键和值
from sklearn.datasets import load_iris
iris_dataset = load_iris()
#print("key of iris_dataset:\n{}".format(iris_dataset.keys()))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import operator
msg = """

K最近邻(kNN，k-NearestNeighbor)分类算法

点击窗口右上角退出

"""


print(msg)


class KNN(object):

    def __init__(self, k=3):
        self.k = k
    def fit(self, x, y):    #fit()函数将x和y传进去
        self.x = x
        self.y = y
    def _square_distance(self, v1, v2):         #计算任意两点之间的距离平方
        return np.sum(np.square(v1 - v2))
    def _vote(self, ys):    #投票
        ys_unique = np.unique(ys)   #ys取唯一值
        vote_dict = {}  #用字典进行操作
        for y in ys:
            if y not in vote_dict.keys():   #y不在当前字典的键里
                vote_dict[y] = 1    #建立k=0的键，值为1
            else:
                vote_dict[y] += 1   #y在当前字典的键里，值加1
            #第一个参数为可迭代的参数，reverse为从大到小排序
            sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
            return sorted_vote_dict[0][0]
    def predict(self, x):       #接收x参数，多行的点数据，每行是一个二维的向量
        y_pred = []
        for i in range(len(x)):
            #得到当前的x[i]和所有的训练样点之间的平方距离，保存于数组当中
            dist_arr = [self._square_distance(x[i], self.x[j]) for j in range(len(self.x))]     #循环内部训练数据方法计算
            sorted_index = np.argsort(dist_arr)     #从小到大排序距离，返回索引
            top_k_index = sorted_index[:self.k]
            y_pred.append(self._vote(ys=self.y[top_k_index]))       #添加当前x和y的预测值
        return np.array(y_pred)
    def score(self, y_true=None, y_pred=None):      #计算推测值的精度
        if y_true is None or y_pred is None:
            y_pred = self.predict(self.x)
            y_true = self.y
        score=0.0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                score += 1
        score /= len(y_true)    #得到正确率
        return score

#train_test_split()方法用来直接分割数据,利用伪随机数生成器将数据集打乱，random_state为种子数
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

#一般画图使用scatter plot 散点图，但是有一个缺点：只能观察2维的数据情况；如果想观察多个特征之间的数据情况，scatter plot并不可行；
#用pair plot 可以观察到任意两个特征之间的关系图（对角线为直方图）；恰巧：pandas的 scatter_matrix函数能画pair plots
#所以，我们先把训练集转换成DataFrame形式，方便画图
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
#数据的好坏直接影响你模型构建成功与否，现实中我们的数据可能存在许多问题（单位不统一，部分数据缺失等）
#所以我们要提前观察下数据集，观察最好的方法就是看图，pandas为我们提供了一个绘制散点图矩阵的函数，叫做scatter_matrix

#参数解释: frame：数据的dataframe,本例为4*150的矩阵; c是颜色,本例中按照y_train的不同来分配不同的颜色;figsize设置图片的尺寸; marker是散点的形状,'o'是圆形,'*'是星形 ;hist_kwds是直方图的相关参数,{'bins':20}是生成包含20个长条的直方图;
#s是大图的尺寸 ; alpha是图的透明度; cmap是colourmap,就是颜色板
grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,marker='o',figsize=(10,10),hist_kwds={'bins':20},s=60,alpha=0.8,cmap='viridis')
#plt.show()

#visualize data
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, marker='.')  #绘制散点图
plt.show()
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='.')
plt.show()

#knn classifier    
#调用KNN分类器
clf = KNN(k=3)
clf.fit(X_train, y_train)

print('train accuracy: {:.3}'.format(clf.score()))

y_test_pred = clf.predict(X_test)
print('test accuracy: {:.3}'.format(clf.score(y_test, y_test_pred)))



