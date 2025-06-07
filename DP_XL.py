import os
import gzip
import numpy as np
import h5py
import matplotlib.pyplot as plt


class model_xl():
    def load_gz_MNIST(self, dir):
        '''
        根据地址读取.gz文件 并转为数组
        Args:
            dir: 文件地址

        Returns:
            np数组
        '''
        assert os.path.exists(dir), "文件位置错误"

        file_class = os.path.basename(dir)
        if "images" in file_class:
            with gzip.open(dir, 'rb') as f:
                file_data = np.frombuffer(f.read(), np.uint8, offset=16)
                file_data = file_data.reshape(-1, 28, 28)  # resize改变原数组 但是reshape不改变原数组 所以得接收一些
                print("{name}的形状为{size}".format(name=file_class, size=file_data.shape))
        elif "labels" in file_class:
            with gzip.open(dir, 'rb') as f:
                file_data = np.frombuffer(f.read(), np.uint8, offset=8)
                print("{name}的形状为{size}".format(name=file_class, size=file_data.shape))
        else:
            print("文件名称错误")
        # 对于图片60000*28*28 或 10000*28*28
        return file_data

    def load_h5_CAT(self):
        train_dataset = h5py.File('datasets/L1W2/train_catvnoncat.h5', "r")

        train_x = np.array(train_dataset["train_set_x"][:])  # your train set features
        train_y = np.array(train_dataset["train_set_y"][:])  # your train set labels

        test_dataset = h5py.File('datasets/L1W2/test_catvnoncat.h5', "r")
        test_x = np.array(test_dataset["test_set_x"][:])  # your test set features
        test_y = np.array(test_dataset["test_set_y"][:])  # your test set labels

        classes = np.array(test_dataset["list_classes"][:])  # the list of classes
        train_y = train_y.reshape((1, train_y.shape[0]))
        test_y = test_y.reshape((1, test_y.shape[0]))
        print("train_x:{train_x}, train_y:{train_y}".format(train_x=train_x.shape, train_y=train_y.shape))
        print("test_x:{test_x}, test_y:{test_y}".format(test_x=test_x.shape, test_y=test_y.shape))
        print(
            "classes:{classes} is {cla0} and {cla1}  ".format(classes=classes.shape, cla0=classes[0], cla1=classes[1]))
        return train_x, train_y, test_x, test_y, classes

    def flatten(self, data, size) -> np.array:
        """
        将数组展平
        Args:
            data:np数组 -1*28*28

        Returns:展平后的数组

        """

        return data.reshape(-1, size).T

    def one_hot(self, data) -> np.array:
        """
        将数据转成one编码
        Args:
            data:

        Returns:

        """

        datalist = np.zeros(shape=(data.__len__(), 10))
        idx = 0
        for i in data:
            datalist[idx, i] = 1
            idx = idx + 1
        return datalist

    def standardization(self, data):

        return data / 255

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        out  = (1 - np.power(x, 2))
        return out

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def sigmoid_derivative(self, x):
        """
        Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
        You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
        Arguments:
        x -- A scalar or numpy array
        Return:
        ds -- Your computed gradient.
        """
        s = self.sigmoid(x)
        ds = s * (1 - s)
        return ds

    def softmax(self, x):
        """Calculates the softmax for each row of the input x.

        Your code should work for a row vector and also for matrices of shape (n, m).

        Argument:
        x -- A numpy matrix of shape (n,m)

        Returns:
        s -- A numpy matrix equal to the softmax of x, of shape (n,m)
        """
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        s = x_exp / x_sum
        return s

    def normalizeRows(self, x):
        """
        Implement a function that normalizes each row of the matrix x (to have unit length).
        Argument:
        x -- A numpy matrix of shape (n, m)

        Returns:
        x -- The normalized (by row) numpy matrix. You are allowed to modify x.
        """
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / x_norm
        return x

    def L1(self, yhat, y):
        """
        Arguments:
        yhat -- vector of size m (predicted labels)
        y -- vector of size m (true labels)

        Returns:
        loss -- the value of the L1 loss function defined above
        """
        loss = np.sum(np.abs(y - yhat))
        return loss

    def L2(self, yhat, y):
        """
        Arguments:
        yhat -- vector of size m (predicted labels)
        y -- vector of size m (true labels)

        Returns:
        loss -- the value of the L2 loss function defined above
        """
        loss = np.dot((y - yhat), (y - yhat).T)
        return loss
    def accuracy(self,x, y):
        acc = 100 - np.mean(np.abs(x - y)) * 100
        return acc

    def cross_entropy(self,A, Y):
        #     j(a,y)=-(y*log(a))-(1-y)*log(1-a) 交叉熵损失

        logprobs = Y * np.log(A) + (1 - Y) * np.log(1 - A)

        return logprobs
    def cross_entropy_derivative(self,A, Y):
        #     j(a,y)=-(y*log(a))-(1-y)*log(1-a) 交叉熵损失
        #     dj/da2 = -y/a + (1-y)/(1-a)  交叉熵的导数
        cost = -1*Y/A+(1-Y)/(1-A)
        return cost

    def relu(self,x):

        return (np.abs(x) + x) / 2.0

    def relu_derivative(self,x):

        return np.where(x > 0, 1, 0)

    def buildnet(self, desentlayer):
        """
        建立全连接网络 创建 权重与偏置
        输入是一个列表 [输入的特征数，第0层的节点数，第1层的节点数，...，输出层的节点数]
        Returns:
        返回w与b的列表 w[0]代表第0层的初始化后的参数 b同理
        """
        inputsize = desentlayer[0]

        w = []
        b = []
        for i in range(1, desentlayer.shape[0]):
            pre = desentlayer[i - 1]
            now = desentlayer[i]
            W = np.random.randn(pre, now) * 0.01
            B = np.zeros((1, now))
            #B = 0
            w.append(W)
            b.append(B)
        return w, b

    def forward(self,x, w, b, act):
        """
        前向传播
        Args:
            x 输入样本
            w: 权重
            b: 偏置
            act: 激活函数的列表 内部存储 ，每一层使用的激活函数

        Returns:
            z代表没有激活的输出
            a代表经过激活的输出
            所谓输出就是矩阵乘法后的值
        """
        assert w.__len__() == b.__len__() == act.__len__(), print("模型不匹配")
        len = w.__len__()
        z = []
        a = []
        for i in range(len):
            Z = np.dot(x, w[i]) + b[i]
            z.append(Z)
            if act[i] == 'tanh':
                A = self.tanh(Z)
                a.append(A)
            elif act[i] == 'sigmoid':
                A = self.sigmoid(Z)
                a.append(A)
            elif act[i] == 'relu':
                A = self.relu(Z)
                a.append(A)
            else:
                print('激活函数设置错误')
            x = A
        return z, a

    def compute_cost(self,A, Y, lossfun):
        """
        计算损失
        Args:
            A: 经过激活函数的输出 然后A = A[-1] 这是指最后一层的输出 也就是预测值
            Y: 真实值
            lossfun: 使用的激活函数 是个字符串

        Returns:
            返回损失
        """
        A = A[-1]
        if lossfun == 'cross-entropy':
            m = Y.shape[0]
            logprobs = self.cross_entropy(A, Y)
            cost = -1 / m * np.sum(logprobs)
            cost = np.squeeze(cost)
        return cost

    def backward(self,w, b, Z, A, X, Y, act, loss):
        """
        反向传播
        Args:
            w: 更新的权重
            b: 更新的偏置
            Z: 没用激活的网络输出
            A:  激活后的网络输出
            X: 输入样本
            Y: 真实值
            act: 激活函数的列表
            loss: 使用的损失函数的字符串

        Returns:
            dw w的梯度的列表 dw[0]代表w[0]的梯度 即第0层网络的权重的梯度
            db 同理
        """
        L = w.__len__()
        m = X.shape[0]
        dZ = [0 for _ in range(L)]
        dw = [0 for _ in range(L)]
        db = [0 for _ in range(L)]
        for i in range(L):
            j = L - i - 1
            if j == L - 1:
                if loss == 'cross-entropy':
                    #dZ[j] = - (np.divide(Y, A[j]) - np.divide(1 - Y, 1 - A[j]))
                    dZ[j] = self.cross_entropy_derivative(A[j], Y)
                else:
                    print("损失函数输入有误")
            else:
                dZ[j] = np.dot(dZ[j + 1], w[j + 1].T)
            if act[j] == 'sigmoid':
                dZ[j] *= self.sigmoid_derivative(Z[j])
            elif act[j] == 'tanh':
                dZ[j] *= self.tanh_derivative(A[j])
            elif act[j] == 'relu':
                dZ[j] *= self.relu_derivative(A[j])
            if j == 0:
                outi_1 = X
            else:
                outi_1 = A[j - 1]
            dw[j] = 1 / m * np.dot(outi_1.T, dZ[j])
            db[j] = 1 / m * np.sum(dZ[j], axis=1, keepdims=True)
        return dw, db

    def update(self,w, b, dw, db, learning_rate=1.2):
        """
        梯度下降函数
        Args:
            w: 更新的权重
            b: 更新的偏置
            dw: w的梯度
            db: 同理
            learning_rate:  学习率

        Returns:
            W w的列表 w[0] 即第0层网络的权重的梯度
            b 同理
        """
        wlist = []
        blist = []
        len = w.__len__()
        for i in range(len):
            W = w[i]
            B = b[i]
            dW = dw[i]
            dB = db[i]
            wlist.append(W - learning_rate * dW)
            blist.append(B - learning_rate * dB)

        return wlist, blist

    def predic(self,x, w, b, act):
        """
        预测
        Args:
            x:输入样本
            w: 更新后的权重
            b: 同理
            act: 使用的各层激活函数的列表

        Returns:
            返回预测的值 实际上与真实值是一样大的数组
        """
        Z, A = self.forward( x, w, b, act)
        out = A[-1]
        prevalue = np.around(out)
        return prevalue
class Example():
    def __init__(self):
        self.mymodel = model_xl()

    def mnist_demo(self):
        """
        手写数字识别
        Returns:

        """
        train_labels = self.mymodel.load_gz_MNIST("./datasets/MNIST/train-labels-idx1-ubyte.gz")
        train_images = self.mymodel.load_gz_MNIST("./datasets/MNIST/train-images-idx3-ubyte.gz")
        test_images = self.mymodel.load_gz_MNIST("./datasets/MNIST/t10k-images-idx3-ubyte.gz")
        test_labels = self.mymodel.load_gz_MNIST("./datasets/MNIST/t10k-labels-idx1-ubyte.gz")

    def cat_demo(self):

        """
        基于sigmoid的进行二分类 即逻辑斯特回归 此方法适合线性可分的案例
        此网络没有设置隐藏层 损失函数采用的是交叉熵损失 也是比较适合二分类
        Returns:

        """
        def propagate(w, b, X, Y, m):

            A = self.mymodel.sigmoid(np.dot(w.T, X) + b)

            cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
            dw = 1 / m * np.dot(X, (A - Y).T)
            db = 1 / m * np.sum(A - Y)
            cost = np.squeeze(cost)  # 删除所有的单维度条目
            grads = {"dw": dw,
                     "db": db}

            return grads, cost

        def Dense(input):
            w = np.zeros((input, 1))
            b = 0
            return w, b

        def optimize(w, b, X, Y, m, num_iterations, learning_rate):
            costs = []
            for i in range(num_iterations):
                grads, cost = propagate(w, b, X, Y, m)

                dw = grads["dw"]
                db = grads["db"]

                w = w - learning_rate * dw
                b = b - learning_rate * db

                costs.append(cost)

            params = {"w": w,
                      "b": b}

            grads = {"dw": dw,
                     "db": db}

            return params, grads, costs

        def predict(w, b, X):
            m = X.shape[1]
            Y_prediction = np.zeros((1, m))
            A = self.mymodel.sigmoid(np.dot(w.T, X) + b)
            for i in range(A.shape[1]):
                if A[0, i] <= 0.5:
                    Y_prediction[0, i] = 0
                else:
                    Y_prediction[0, i] = 1

            return Y_prediction



        def show(index):
            plt.imshow(train_x[index])
            plt.title(str(index) + " is " + str(classes[train_y[0][index]]))
            plt.show()

        train_x, train_y, test_x, test_y, classes = self.mymodel.load_h5_CAT()
        show(25)

        train_x_flatten = self.mymodel.flatten(train_x, 64 * 64 * 3)
        test_x_flatten = self.mymodel.flatten(test_x, 64 * 64 * 3)

        train_set_x = self.mymodel.standardization(train_x_flatten)
        test_set_x = self.mymodel.standardization(test_x_flatten)

        w, b = Dense(64 * 64 * 3)
        # grads, cost = propagate(w, b, train_set_x,train_y,train_set_x.shape[1])
        params, grads, costs = optimize(w, b, train_set_x, train_y, train_set_x.shape[1], num_iterations=1000,
                                        learning_rate=0.009)

        predictions = predict(params["w"], params["b"], test_set_x)
        acc = self.mymodel.accuracy(predictions, test_y)
        print(acc)
        costs = np.squeeze(costs)
        plt.plot(costs)
        plt.show()

    def flower_demo(self):
        """
        这是自己实现花形状的分类 使用了一个隐藏层 两种激活函数 交叉熵损失
        Returns:

        """
        def load_planar_dataset():
            np.random.seed(1)
            m = 400  # number of examples
            N = int(m / 2)  # number of points per class
            D = 2  # dimensionality
            X = np.zeros((m, D))  # data matrix where each row is a single example
            Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
            a = 4  # maximum ray of the flower

            for j in range(2):
                ix = range(N * j, N * (j + 1))
                t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
                r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
                X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
                Y[ix] = j

            X = X.T
            Y = Y.T
            idx = 0
            for i in Y[0]:
                plt.scatter(X[0, idx], X[1, idx], c='red') if i == 1 else plt.scatter(X[0, idx], X[1, idx], c='blue')
                idx += 1
            plt.show()

            return X, Y
        def buildnet(desentlayer):
            '''
            2,4,1  x,h,y 样本数是400 输入是（2，400） 转置 （400，2）
            第一层（400，2）*（2，4） = （400，4） + （1，4） = （400，4） 激活
            第二层 （400，4）*（4，1） = （400，1）+（1,1） = (400,1)

            x 2,400  w1 4,2 b1 4,1 w2 1,4 b2 1,1
            z1 = w1*x+b  a1 = tanh(z1)  z2 = w2*a1+b2  a2 = tanh(z2)
            (4,400) = (4,2)*(2,400) +(4,1) (1,400) = (1,4)*(4,400)+(1,1)
            Args:
                desentlayer:

            Returns:

            '''
            inputsize = desentlayer[0]
            """
             2,4,1  x,h,y 样本数是400 输入是（2，400） 转置 （400，2）
            第一层（400，2）*（2，4） = （400，4） + （1，4） = （400，4） 激活
            第二层 （400，4）*（4，1） = （400，1）+（1,1） = (400,1)
            """
            w = []
            b = []
            for i in range(1, desentlayer.shape[0]):
                pre = desentlayer[i-1]
                now = desentlayer[i]
                W = np.random.randn(pre, now) * 0.01
                B = np.zeros(( 1,now))
                w.append(W)
                b.append(B)
            return w,b
        def forward(x,w,b,act):
            assert w.__len__()==b.__len__()==act.__len__(),print("模型不匹配")

            len = w.__len__()
            z = []
            a = []
            for i in range(len):
                Z = np.dot(x,w[i]) + b[i]
                z.append(Z)
                if act[i] == 'tanh':
                    A = self.mymodel.tanh(Z)
                    a.append(A)
                elif act[i] == 'sigmoid':
                    A = self.mymodel.sigmoid(Z)
                    a.append(A)
                else:print('激活函数设置错误')
                x = A
            return z,a

        def compute_cost(A, Y,lossfun):
            A = A[-1]
            if lossfun=='cross-entropy':
                m = Y.shape[0]
                logprobs = self.mymodel.cross_entropy(A,Y)
                cost = -1 / m * np.sum(logprobs)
                cost = np.squeeze(cost)
            return cost

        def backward(w,b,Z,A,X,Y):

            """
            j(a,y)=-(y*log(a))-(1-y)*log(1-a) 交叉熵损失
            dj/dw2 = dj/dz2 * dz2/dw2
            dj/db2 = dj/dz2 * dz2/db2
            第一步 链式法则
            dj/dz2 = dj/da2  *  da2/dz2  = a2-y
            dj/da2 = -y/a + (1-y)/(1-a)  交叉熵的导数
            da2/dz2 = a(a-1)  sigmoid 导数
            第二步
            z2 = a1*w + b
            dz2/dw2 = a1
            dz2/db2  = 1
            结合
            dj/dw2 =（ a2-y ）*a1   400，4 * 400，1  4,1
            dj/db2 =（ a2-y ）*1
            最后 由于w 4，1 ,b 1，1  如果有m个样本 那么就有m个w,b的梯度 所以要平均

            第一步 链式法则  z2 = w*a1+b
            dj/dz1 = dj/z2 * dz2/da1  * da1/dz1
                   =  （ a2-y ）* w2   * da1/dz1
            z2 = w2*a1+b
            dz2/da1 = w2
            a1 = tanh(z1)
            da1/dz1 = 1-tanh(z1)*tanh(z1)
            第二步
            z1 = x*w + b
            dz1/dw1 = x
            dz1/db1  = 1
            结合
            dj/dw1 =（ a2-y ）* w2*1-tanh(z1)*tanh(z1)*x
            dj/db1 =（ a2-y ）* w2*1-tanh(z1)*tanh(z1)*1
            最后 由于w 4，1 ,b 1，1  如果有m个样本 那么就有m个w,b的梯度 所以要平均
            """

            m = X.shape[0]
            W1 = w[0]
            W2 = w[1]
            A1 = A[0]
            A2 = A[1]
            Z1 = Z[0]
            Z2 = Z[1]
            dZ2 = self.mymodel.cross_entropy_derivative(A2, Y) * self.mymodel.sigmoid_derivative(Z2)
            dW2 = 1 / m * np.dot(A1.T, dZ2)  # 4 ,1
            db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

            dZ1 = np.dot(dZ2, W2.T) * self.mymodel.tanh_derivative(A1)
            dW1 = 1 / m * np.dot(X.T, dZ1)  # 2,4
            db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

            dw = []
            db = []
            dw.append(dW1)
            dw.append(dW2)
            db.append(db1)
            db.append(db2)

            return dw,db

        def backward_absorb(w,b,Z,A,X,Y,act,loss):
            L = w.__len__()
            m = X.shape[0]
            dZ = [0 for _ in range(L )]
            dw = [0 for _ in range(L )]
            db = [0 for _ in range(L )]
            for i in range(L):
                j = L-i-1
                if j==L-1:
                    if loss=='cross-entropy':
                        dZ[j] = self.mymodel.cross_entropy_derivative(A[j], Y)
                    else:print("损失函数输入有误")
                else:
                    dZ[j] = np.dot(dZ[j+1], w[j+1].T)
                if act[j] =='sigmoid':
                    dZ[j] *= self.mymodel.sigmoid_derivative(Z[j])
                elif act[j] =='tanh':
                    dZ[j] *= self.mymodel.tanh_derivative(A[j])
                if j==0:outi_1 = X
                else:outi_1 = A[j-1]
                dw[j] = 1 / m * np.dot(outi_1.T, dZ[j])  # 4 ,1
                db[j]  = 1 / m * np.sum(dZ[j] , axis=1, keepdims=True)
            return dw,db

        def update(w, b,dw,db, learning_rate=1.2):

            wlist = []
            blist = []
            len = w.__len__()
            for i in range(len):

                W = w[i]
                B = b[i]
                dW = dw[i]
                dB = db[i]
                wlist.append(W - learning_rate * dW)
                blist.append(B - learning_rate * dB)
            return wlist, blist

        def predic(x, w, b, act):
            Z,A = forward(x, w, b, act)
            out  = A[-1]
            prevalue = np.around(out)
            return prevalue


        epoch = 500
        X, Y = load_planar_dataset()
        X = X.T
        Y = Y.T
        w,b = buildnet(np.array([2,4,1]))
        act = ['tanh','sigmoid']
        loss = 'cross-entropy'
        costlist = []
        for i in range(epoch):
            Z,A = forward(X,w,b,act)
            cost = compute_cost(A, Y,loss)
            dw,db = backward(w,b,Z,A,X,Y)
            #dw,db = backward_absorb(w,b,Z,A,X,Y,act,loss)
            w,b = update(w,b,dw,db)
            costlist.append(cost)

        predic = predic(X,w,b,act)
        acc = self.mymodel.accuracy(predic, Y)
        print(acc)
        plt.plot(costlist)
        plt.show()

        idx = 0
        predic = predic.T
        X = X.T
        for i in predic[0]:
            plt.scatter(X[0, idx], X[1, idx], c='red') if i == 1 else plt.scatter(X[0, idx], X[1, idx], c='blue')
            idx += 1
        plt.show()

    def cat_new_demo(self):
        """
        #自己编写实现 预测功能不完善
        这也是基于猫数据的分类 不同的是重新设计函数 使得可以任意设置网络的层数 与激活函数
        Returns:

        """

        def show(index):
            plt.imshow(train_x[index])
            plt.title(str(index) + " is " + str(classes[train_y[0][index]]))
            plt.show()


        train_x, train_y, test_x, test_y, classes = self.mymodel.load_h5_CAT()
        show(25)
        train_x_flatten = self.mymodel.flatten(train_x, 64 * 64 * 3)
        test_x_flatten = self.mymodel.flatten(test_x, 64 * 64 * 3)
        train_set_x = self.mymodel.standardization(train_x_flatten).T
        test_set_x = self.mymodel.standardization(test_x_flatten).T
        train_set_y = train_y.T
        test_set_y = test_y.T
        epoch = 1000
        batch_size = test_set_x.shape[0]

        w, b = self.mymodel.buildnet( np.array([64*64*3, 2, 1]))
        act = ['relu', 'sigmoid']
        loss = 'cross-entropy'
        costlist = []

        for i in range(epoch):
            train_x = train_set_x
            train_y = train_set_y
            Z, A = self.mymodel.forward(train_x, w, b, act)
            cost = self.mymodel.compute_cost(A, train_y, loss)
            dw, db = self.mymodel.backward(w, b, Z, A, train_x, train_y, act, loss)
            w, b = self.mymodel.update(w, b, dw, db)
            costlist.append(cost)
        predic = self.mymodel.predic(train_set_x, w, b, act)
        acc = self.mymodel.accuracy(predic,train_set_y)
        print(acc)
        plt.plot(costlist)
        plt.show()
        for i in range(costlist.__len__()):
            if i%50==0:
                print("第{i}次 损失为{loss}".format(i=i, loss=costlist[i]))##

    def cat_final_demo(self):
        """
        #课程代码
        和上面的是一样的
        Returns:

        """

        def sigmoid(Z):
            """
            Implements the sigmoid activation in numpy

            Arguments:
            Z -- numpy array of any shape

            Returns:
            A -- output of sigmoid(z), same shape as Z
            cache -- returns Z as well, useful during backpropagation
            """

            A = 1 / (1 + np.exp(-Z))
            cache = Z

            return A, cache

        def sigmoid_backward(dA, cache):
            """
            Implement the backward propagation for a single SIGMOID unit.

            Arguments:
            dA -- post-activation gradient, of any shape
            cache -- 'Z' where we store for computing backward propagation efficiently

            Returns:
            dZ -- Gradient of the cost with respect to Z
            """

            Z = cache

            s = 1 / (1 + np.exp(-Z))
            dZ = dA * s * (1 - s)

            assert (dZ.shape == Z.shape)

            return dZ

        def relu(Z):
            """
            Implement the RELU function.

            Arguments:
            Z -- Output of the linear layer, of any shape

            Returns:
            A -- Post-activation parameter, of the same shape as Z
            cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
            """

            A = np.maximum(0, Z)

            assert (A.shape == Z.shape)

            cache = Z
            return A, cache

        def relu_backward(dA, cache):
            """
            Implement the backward propagation for a single RELU unit.

            Arguments:
            dA -- post-activation gradient, of any shape
            cache -- 'Z' where we store for computing backward propagation efficiently

            Returns:
            dZ -- Gradient of the cost with respect to Z
            """

            Z = cache
            dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

            # When z <= 0, you should set dz to 0 as well.
            dZ[Z <= 0] = 0

            assert (dZ.shape == Z.shape)

            return dZ

        def initialize_parameters(layer_dims):
            """
            Arguments:
            layer_dims -- python array (list) containing the dimensions of each layer in our network

            Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                            bl -- bias vector of shape (layer_dims[l], 1)
                              20, 7, 5, 1
            (12288, 209)  w1 ( 20 12288）  20 5
            """

            np.random.seed(3)
            parameters = {}
            L = len(layer_dims)
            for l in range(1, L):
                parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) *np.sqrt(2./layers_dims[l-1])#he方法初始化 防止梯度消失
                parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

                assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
                assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

            return parameters

        def linear_forward(A, W, b):
            """
            Implement the linear part of a layer's forward propagation.

            Arguments:
            A -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)

            Returns:
            Z -- the input of the activation function, also called pre-activation parameter
            cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
            """

            Z = np.dot(W, A) + b

            assert (Z.shape == (W.shape[0], A.shape[1]))
            cache = (A, W, b)

            return Z, cache

        def linear_activation_forward(A_prev, W, b, activation):
            """
            Implement the forward propagation for the LINEAR->ACTIVATION layer

            Arguments:
            A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
            W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
            b -- bias vector, numpy array of shape (size of the current layer, 1)
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

            Returns:
            A -- the output of the activation function, also called the post-activation value
            cache --  A W B Z

            """

            if activation == "sigmoid":
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = sigmoid(Z)
            elif activation == "relu":
                Z, linear_cache = linear_forward(A_prev, W, b)
                A, activation_cache = relu(Z)
            assert (A.shape == (W.shape[0], A_prev.shape[1]))
            cache = (linear_cache, activation_cache)
            return A, cache

        def forward(X, parameters):
            """
            Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

            Arguments:
            X -- data, numpy array of shape (input size, number of examples)
            parameters --w b

            Returns:
            AL -- 最后一层的输出
            caches --  各层的 a w b z


            """

            caches = []
            A = X
            L = len(parameters)//2
            for l in range(1, L):
                A_prev = A
                A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
                caches.append(cache)

            AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
            caches.append(cache)
            assert (AL.shape == (1, X.shape[1]))

            return AL, caches

        def compute_cost(AL, Y):
            """
            Implement the cost function defined by equation (7).

            Arguments:
            AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
            Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

            Returns:
            cost -- cross-entropy cost
            """

            m = Y.shape[1]

            cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis=1, keepdims=True)

            cost = np.squeeze( cost)
            assert (cost.shape == ())

            return cost

        def linear_backward(dZ, cache):
            """
            Implement the linear portion of backward propagation for a single layer (layer l)

            Arguments:
            dZ -- Gradient of the cost with respect to the linear output (of current layer l)
            cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

            Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
            """
            A_prev, W, b = cache
            m = A_prev.shape[1]

            dW = 1 / m * np.dot(dZ, A_prev.T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)

            assert (dA_prev.shape == A_prev.shape)
            assert (dW.shape == W.shape)
            assert (db.shape == b.shape)

            return dA_prev, dW, db

        def activation_backward(dA, cache, activation):
            """
            Implement the backward propagation for the LINEAR->ACTIVATION layer.

            Arguments:
            dA -- post-activation gradient for current layer l
            cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
            activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

            Returns:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
            """
            linear_cache, activation_cache = cache

            if activation == "relu":

                dZ = relu_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)

            elif activation == "sigmoid":

                dZ = sigmoid_backward(dA, activation_cache)
                dA_prev, dW, db = linear_backward(dZ, linear_cache)

            return dA_prev, dW, db

        def backward(AL, Y, caches):
            """
            Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

            Arguments:
            AL -- probability vector, output of the forward propagation (L_model_forward())
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
            caches -- list of caches containing:
                        every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                        the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

            Returns:
            grads -- A dictionary with the gradients
                     grads["dA" + str(l)] = ...
                     grads["dW" + str(l)] = ...
                     grads["db" + str(l)] = ...
            """
            grads = {}
            L = len(caches)
            m = AL.shape[1]
            Y = Y.reshape(AL.shape)

            #采用交叉熵计算损失 这是损失函数的梯度
            dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

            current_cache = caches[L - 1]
            grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, activation="sigmoid")

            for l in reversed(range(L - 1)):

                current_cache = caches[l]
                dA_prev_temp, dW_temp, db_temp =  activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
                grads["dA" + str(l + 1)] = dA_prev_temp
                grads["dW" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp

            return grads

        def update_parameters(parameters, grads, learning_rate):
            """
            Update parameters using gradient descent

            Arguments:
            parameters -- python dictionary containing your parameters
            grads -- python dictionary containing your gradients, output of L_model_backward

            Returns:
            parameters -- python dictionary containing your updated parameters
                          parameters["W" + str(l)] = ...
                          parameters["b" + str(l)] = ...
            """

            L = len(parameters) // 2

            for l in range(L):
                parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
                parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

            return parameters

        def predict(X, y, parameters):
            """
            该函数用于预测L层神经网络的结果，当然也包含两层

            参数：
             X - 测试集
             y - 标签
             parameters - 训练模型的参数

            返回：
             p - 给定数据集X的预测
            """

            m = X.shape[1]
            n = len(parameters) // 2  # 神经网络的层数
            p = np.zeros((1, m))

            probas, caches =  forward(X, parameters)

            for i in range(0, probas.shape[1]):
                if probas[0, i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0

            print("准确度为: " + str(float(np.sum((p == y)) / m)))

            return p

        def  model(X, Y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=False):  # lr was 0.009
            """
            Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

            Arguments:
            X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
            layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
            learning_rate -- learning rate of the gradient descent update rule
            num_iterations -- number of iterations of the optimization loop
            print_cost -- if True, it prints the cost every 100 steps

            Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
            """

            np.random.seed(1)
            costs = []

            parameters = initialize_parameters(layers_dims)

            for i in range(0, num_iterations):

                AL, caches =  forward(X, parameters)

                cost = compute_cost(AL, Y)

                grads = backward(AL, Y, caches)

                parameters = update_parameters(parameters, grads, learning_rate)

                if print_cost and i % 100 == 0:
                    print("Cost after iteration %i: %f" % (i, cost))
                if print_cost and i % 100 == 0:
                    costs.append(cost)

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            return parameters

        train_x_orig, train_y, test_x_orig, test_y, classes = self.mymodel.load_h5_CAT()
        train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
        test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

        train_x = train_x_flatten / 255.
        test_x = test_x_flatten / 255.

        layers_dims = [64*64*3, 20, 7, 5, 1]
        parameters =  model(train_x, train_y, layers_dims, num_iterations=1000, print_cost=True)

        pred_train = predict(train_x, train_y, parameters)
        pred_test = predict(test_x, test_y, parameters)

    def init_demo(self):
        """
        用来对比不同的初始化参数方法对训练的影响
        Returns:
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn
        import sklearn.datasets

        def sigmoid(x):
            """
            Compute the sigmoid of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- sigmoid(x)
            """
            s = 1 / (1 + np.exp(-x))
            return s

        def relu(x):
            """
            Compute the relu of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- relu(x)
            """
            s = np.maximum(0, x)

            return s

        def compute_loss(a3, Y):

            """
            Implement the loss function

            Arguments:
            a3 -- post-activation, output of forward propagation
            Y -- "true" labels vector, same shape as a3

            Returns:
            loss - value of the loss function
            """

            m = Y.shape[1]
            logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
            loss = 1. / m * np.nansum(logprobs)

            return loss

        def forward_propagation(X, parameters):
            """
            Implements the forward propagation (and computes the loss) presented in Figure 2.

            Arguments:
            X -- input dataset, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                            W1 -- weight matrix of shape ()
                            b1 -- bias vector of shape ()
                            W2 -- weight matrix of shape ()
                            b2 -- bias vector of shape ()
                            W3 -- weight matrix of shape ()
                            b3 -- bias vector of shape ()

            Returns:
            loss -- the loss function (vanilla logistic loss)
            """

            # retrieve parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]

            # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
            z1 = np.dot(W1, X) + b1
            a1 = relu(z1)
            z2 = np.dot(W2, a1) + b2
            a2 = relu(z2)
            z3 = np.dot(W3, a2) + b3
            a3 = sigmoid(z3)

            cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

            return a3, cache

        def backward_propagation(X, Y, cache):
            """
            Implement the backward propagation presented in figure 2.

            Arguments:
            X -- input dataset, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
            cache -- cache output from forward_propagation()

            Returns:
            gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
            """
            m = X.shape[1]
            (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

            dz3 = 1. / m * (a3 - Y)
            dW3 = np.dot(dz3, a2.T)
            db3 = np.sum(dz3, axis=1, keepdims=True)

            da2 = np.dot(W3.T, dz3)
            dz2 = np.multiply(da2, np.int64(a2 > 0))
            dW2 = np.dot(dz2, a1.T)
            db2 = np.sum(dz2, axis=1, keepdims=True)

            da1 = np.dot(W2.T, dz2)
            dz1 = np.multiply(da1, np.int64(a1 > 0))
            dW1 = np.dot(dz1, X.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)

            gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                         "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                         "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

            return gradients

        def update_parameters(parameters, grads, learning_rate):
            """
            Update parameters using gradient descent

            Arguments:
            parameters -- python dictionary containing your parameters
            grads -- python dictionary containing your gradients, output of n_model_backward

            Returns:
            parameters -- python dictionary containing your updated parameters
                          parameters['W' + str(i)] = ...
                          parameters['b' + str(i)] = ...
            """

            L = len(parameters) // 2  # number of layers in the neural networks

            # Update rule for each parameter
            for k in range(L):
                parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
                parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

            return parameters

        def predict(X, y, parameters):
            """
            This function is used to predict the results of a  n-layer neural network.

            Arguments:
            X -- data set of examples you would like to label
            parameters -- parameters of the trained model

            Returns:
            p -- predictions for the given dataset X
            """

            m = X.shape[1]
            p = np.zeros((1, m) )

            # Forward propagation
            a3, caches = forward_propagation(X, parameters)

            # convert probas to 0/1 predictions
            for i in range(0, a3.shape[1]):
                if a3[0, i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0

            # print results
            print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

            return p

        def load_dataset():
            np.random.seed(1)
            train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
            np.random.seed(2)
            test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
            # Visualize the data

            plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
            plt.show()
            train_X = train_X.T
            train_Y = train_Y.reshape((1, train_Y.shape[0]))
            test_X = test_X.T
            test_Y = test_Y.reshape((1, test_Y.shape[0]))
            return train_X, train_Y, test_X, test_Y

        def plot_decision_boundary(model, X, y):
            # Set min and max values and give it some padding
            x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
            y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
            h = 0.01
            # Generate a grid of points with distance h between them
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            # Predict the function value for the whole grid
            Z = model(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            # Plot the contour and training examples
            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
            plt.ylabel('x2')
            plt.xlabel('x1')
            plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
            plt.show()

        def predict_dec(parameters, X):
            """
            Used for plotting decision boundary.

            Arguments:
            parameters -- python dictionary containing your parameters
            X -- input data of size (m, K)

            Returns
            predictions -- vector of predictions of our model (red: 0 / blue: 1)
            """

            # Predict using forward propagation and a classification threshold of 0.5
            a3, cache = forward_propagation(X, parameters)
            predictions = (a3 > 0.5)
            return predictions

        def initialize_parameters_zeros(layers_dims):
            """
            将模型的参数全部设置为0

            参数：
                layers_dims - 列表，模型的层数和对应每一层的节点的数量
            返回
                parameters - 包含了所有W和b的字典
                    W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
                    b1 - 偏置向量，维度为（layers_dims[1],1）
                    ···
                    WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
                    bL - 偏置向量，维度为（layers_dims[L],1）
            """
            parameters = {}

            L = len(layers_dims)  # 网络层数

            for l in range(1, L):
                parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
                parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

                # 使用断言确保我的数据格式是正确的
                assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
                assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

            return parameters

        def initialize_parameters_random(layers_dims):
            """
            参数：
                layers_dims - 列表，模型的层数和对应每一层的节点的数量
            返回
                parameters - 包含了所有W和b的字典
                    W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
                    b1 - 偏置向量，维度为（layers_dims[1],1）
                    ···
                    WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
                    b1 - 偏置向量，维度为（layers_dims[L],1）
            """

            np.random.seed(3)  # 指定随机种子
            parameters = {}
            L = len(layers_dims)  # 层数

            for l in range(1, L):
                parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10  # 使用10倍缩放
                parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

                # 使用断言确保我的数据格式是正确的
                assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
                assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

            return parameters

        def initialize_parameters_he(layers_dims):
            """
            参数：
                layers_dims - 列表，模型的层数和对应每一层的节点的数量
            返回
                parameters - 包含了所有W和b的字典
                    W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
                    b1 - 偏置向量，维度为（layers_dims[1],1）
                    ···
                    WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
                    b1 - 偏置向量，维度为（layers_dims[L],1）
            """

            np.random.seed(3)  # 指定随机种子
            parameters = {}
            L = len(layers_dims)  # 层数

            for l in range(1, L):
                parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
                    2 / layers_dims[l - 1])
                parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

                # 使用断言确保我的数据格式是正确的
                assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
                assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

            return parameters


        def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he", is_polt=True):
            """
            实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

            参数：
                X - 输入的数据，维度为(2, 要训练/测试的数量)
                Y - 标签，【0 | 1】，维度为(1，对应的是输入的数据的标签)
                learning_rate - 学习速率
                num_iterations - 迭代的次数
                print_cost - 是否打印成本值，每迭代1000次打印一次
                initialization - 字符串类型，初始化的类型【"zeros" | "random" | "he"】
                is_polt - 是否绘制梯度下降的曲线图
            返回
                parameters - 学习后的参数
            """
            grads = {}
            costs = []
            m = X.shape[1]
            layers_dims = [X.shape[0], 10, 5, 1]

            # 选择初始化参数的类型
            if initialization == "zeros":
                parameters = initialize_parameters_zeros(layers_dims)
            elif initialization == "random":
                parameters = initialize_parameters_random(layers_dims)
            elif initialization == "he":
                parameters = initialize_parameters_he(layers_dims)
            else:
                print("错误的初始化参数！程序退出")
                exit

            # 开始学习
            for i in range(0, num_iterations):
                # 前向传播
                a3, cache =  forward_propagation(X, parameters)

                # 计算成本
                cost =  compute_loss(a3, Y)

                # 反向传播
                grads =  backward_propagation(X, Y, cache)

                # 更新参数
                parameters =  update_parameters(parameters, grads, learning_rate)

                # 记录成本
                if i % 1000 == 0:
                    costs.append(cost)
                    # 打印成本
                    if print_cost:
                        print("第" + str(i) + "次迭代，成本值为：" + str(cost))

            # 学习完毕，绘制成本曲线
            if is_polt:
                plt.plot(costs)
                plt.ylabel('cost')
                plt.xlabel('iterations (per hundreds)')
                plt.title("Learning rate =" + str(learning_rate))
                plt.show()

            # 返回学习完毕后的参数
            return parameters

        train_X, train_Y, test_X, test_Y = load_dataset( )

        #parameters = model(train_X, train_Y, initialization="zeros", is_polt=True)
        #parameters = model(train_X, train_Y, initialization="random", is_polt=True)
        parameters = model(train_X, train_Y, initialization="he", is_polt=True)
        print("训练集:")
        predictions_train =  predict(train_X, train_Y, parameters)
        print("测试集:")
        predictions_test =  predict(test_X, test_Y, parameters)

        print("predictions_train = " + str(predictions_train))
        print("predictions_test = " + str(predictions_test))

        axes = plt.gca()
        axes.set_xlim([-1.5, 1.5])
        axes.set_ylim([-1.5, 1.5])
        plot_decision_boundary(lambda x:  predict_dec(parameters, x.T), train_X, train_Y)

    def reg_demo(self):
        """
        这个案例来探究正则化对模型的影响

        问题描述：假设你现在是一个AI专家，你需要设计一个模型，可以用于推荐在足球场中守门员将球发至哪个位置可以让本队的球员抢到球的可能性更大。
        说白了，实际上就是一个二分类，一半是己方抢到球，一半就是对方抢到球

        Returns:

        """

        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.io as sio

        def sigmoid(x):
            """
            Compute the sigmoid of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- sigmoid(x)
            """
            s = 1 / (1 + np.exp(-x))
            return s

        def relu(x):
            """
            Compute the relu of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- relu(x)
            """
            s = np.maximum(0, x)

            return s

        def initialize_parameters(layer_dims):
            """
            Arguments:
            layer_dims -- python array (list) containing the dimensions of each layer in our network

            Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                            b1 -- bias vector of shape (layer_dims[l], 1)
                            Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                            bl -- bias vector of shape (1, layer_dims[l])

            Tips:
            - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1].
            This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
            - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
            """

            np.random.seed(3)
            parameters = {}
            L = len(layer_dims)  # number of layers in the network

            for l in range(1, L):
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(
                    layer_dims[l - 1])
                parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

                #assert (parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l - 1])
                #assert (parameters['W' + str(l)].shape == layer_dims[l], 1)

            return parameters

        def forward_propagation(X, parameters):
            """
            Implements the forward propagation (and computes the loss) presented in Figure 2.

            Arguments:
            X -- input dataset, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                            W1 -- weight matrix of shape ()
                            b1 -- bias vector of shape ()
                            W2 -- weight matrix of shape ()
                            b2 -- bias vector of shape ()
                            W3 -- weight matrix of shape ()
                            b3 -- bias vector of shape ()

            Returns:
            loss -- the loss function (vanilla logistic loss)
            """

            # retrieve parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]

            # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
            z1 = np.dot(W1, X) + b1
            a1 = relu(z1)
            z2 = np.dot(W2, a1) + b2
            a2 = relu(z2)
            z3 = np.dot(W3, a2) + b3
            a3 = sigmoid(z3)

            cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

            return a3, cache

        def compute_cost(a3, Y):
            """
            Implement the cost function

            Arguments:
            a3 -- post-activation, output of forward propagation
            Y -- "true" labels vector, same shape as a3

            Returns:
            cost - value of the cost function
            """
            m = Y.shape[1]

            logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
            cost = 1. / m * np.nansum(logprobs)

            return cost

        def backward_propagation(X, Y, cache):
            """
            Implement the backward propagation presented in figure 2.

            Arguments:
            X -- input dataset, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
            cache -- cache output from forward_propagation()

            Returns:
            gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
            """
            m = X.shape[1]
            (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

            dz3 = 1. / m * (a3 - Y)
            dW3 = np.dot(dz3, a2.T)
            db3 = np.sum(dz3, axis=1, keepdims=True)

            da2 = np.dot(W3.T, dz3)
            dz2 = np.multiply(da2, np.int64(a2 > 0))
            dW2 = np.dot(dz2, a1.T)
            db2 = np.sum(dz2, axis=1, keepdims=True)

            da1 = np.dot(W2.T, dz2)
            dz1 = np.multiply(da1, np.int64(a1 > 0))
            dW1 = np.dot(dz1, X.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)

            gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                         "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                         "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

            return gradients

        def update_parameters(parameters, grads, learning_rate):
            """
            Update parameters using gradient descent

            Arguments:
            parameters -- python dictionary containing your parameters
            grads -- python dictionary containing your gradients, output of n_model_backward

            Returns:
            parameters -- python dictionary containing your updated parameters
                          parameters['W' + str(i)] = ...
                          parameters['b' + str(i)] = ...
            """

            L = len(parameters) // 2  # number of layers in the neural networks

            # Update rule for each parameter
            for k in range(L):
                parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
                parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

            return parameters

        def load_2D_dataset(is_plot=True):
            data = sio.loadmat('./datasets/L2W1/datasets/data.mat')
            train_X = data['X'].T
            train_Y = data['y'].T
            test_X = data['Xval'].T
            test_Y = data['yval'].T
            if is_plot:
                plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
                plt.show()
            return train_X, train_Y, test_X, test_Y

        def predict(X, y, parameters):
            """
            This function is used to predict the results of a  n-layer neural network.

            Arguments:
            X -- data set of examples you would like to label
            parameters -- parameters of the trained model

            Returns:
            p -- predictions for the given dataset X
            """

            m = X.shape[1]
            p = np.zeros((1, m) )

            # Forward propagation
            a3, caches = forward_propagation(X, parameters)

            # convert probas to 0/1 predictions
            for i in range(0, a3.shape[1]):
                if a3[0, i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0

            # print results
            print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

            return p

        def plot_decision_boundary(model, X, y):
            # Set min and max values and give it some padding
            x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
            y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
            h = 0.01
            # Generate a grid of points with distance h between them
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            # Predict the function value for the whole grid
            Z = model(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            # Plot the contour and training examples
            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
            plt.ylabel('x2')
            plt.xlabel('x1')
            plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
            plt.show()

        def predict_dec(parameters, X):
            """
            Used for plotting decision boundary.

            Arguments:
            parameters -- python dictionary containing your parameters
            X -- input data of size (m, K)

            Returns
            predictions -- vector of predictions of our model (red: 0 / blue: 1)
            """

            # Predict using forward propagation and a classification threshold of 0.5
            a3, cache = forward_propagation(X, parameters)
            predictions = (a3 > 0.5)
            return predictions

        def compute_cost_with_regularization(A3, Y, parameters, lambd):
            """
            实现公式2的L2正则化计算成本

            参数：
                A3 - 正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
                Y - 标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
                parameters - 包含模型学习后的参数的字典
            返回：
                cost - 使用公式2计算出来的正则化损失的值

            """
            m = Y.shape[1]
            W1 = parameters["W1"]
            W2 = parameters["W2"]
            W3 = parameters["W3"]

            cross_entropy_cost =  compute_cost(A3, Y)

            L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (
                        2 * m)

            cost = cross_entropy_cost + L2_regularization_cost

            return cost

        # 当然，因为改变了成本函数，我们也必须改变向后传播的函数， 所有的梯度都必须根据这个新的成本值来计算。

        def backward_propagation_with_regularization(X, Y, cache, lambd):
            """
            实现我们添加了L2正则化的模型的后向传播。

            参数：
                X - 输入数据集，维度为（输入节点数量，数据集里面的数量）
                Y - 标签，维度为（输出节点数量，数据集里面的数量）
                cache - 来自forward_propagation（）的cache输出
                lambda - regularization超参数，实数

            返回：
                gradients - 一个包含了每个参数、激活值和预激活值变量的梯度的字典
            """

            m = X.shape[1]

            (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

            dZ3 = A3 - Y

            dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
            db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

            dA2 = np.dot(W3.T, dZ3)
            dZ2 = np.multiply(dA2, np.int64(A2 > 0))
            dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
            db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.dot(W2.T, dZ2)
            dZ1 = np.multiply(dA1, np.int64(A1 > 0))
            dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
            db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

            gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                         "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                         "dZ1": dZ1, "dW1": dW1, "db1": db1}

            return gradients

        def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
            """
            实现具有随机舍弃节点的前向传播。
            LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

            参数：
                X  - 输入数据集，维度为（2，示例数）
                parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
                    W1  - 权重矩阵，维度为（20,2）
                    b1  - 偏向量，维度为（20,1）
                    W2  - 权重矩阵，维度为（3,20）
                    b2  - 偏向量，维度为（3,1）
                    W3  - 权重矩阵，维度为（1,3）
                    b3  - 偏向量，维度为（1,1）
                keep_prob  - 随机删除的概率，实数
            返回：
                A3  - 最后的激活值，维度为（1,1），正向传播的输出
                cache - 存储了一些用于计算反向传播的数值的元组
            """
            np.random.seed(1)

            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]

            # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
            Z1 = np.dot(W1, X) + b1
            A1 =  relu(Z1)

            # 下面的步骤1-4对应于上述的步骤1-4。
            D1 = np.random.rand(A1.shape[0], A1.shape[1])  # 步骤1：初始化矩阵D1 = np.random.rand(..., ...)
            D1 = D1 < keep_prob  # 步骤2：将D1的值转换为0或1（使​​用keep_prob作为阈值）
            A1 = A1 * D1  # 步骤3：舍弃A1的一些节点（将它的值变为0或False）
            A1 = A1 / keep_prob  # 步骤4：缩放未舍弃的节点(不为0)的值
            """
            #不理解的同学运行一下下面代码就知道了。
            import numpy as np
            np.random.seed(1)
            A1 = np.random.randn(1,3)

            D1 = np.random.rand(A1.shape[0],A1.shape[1])
            keep_prob=0.5
            D1 = D1 < keep_prob
            print(D1)

            A1 = 0.01
            A1 = A1 * D1
            A1 = A1 / keep_prob
            print(A1)
            """

            Z2 = np.dot(W2, A1) + b2
            A2 =  relu(Z2)

            # 下面的步骤1-4对应于上述的步骤1-4。
            D2 = np.random.rand(A2.shape[0], A2.shape[1])  # 步骤1：初始化矩阵D2 = np.random.rand(..., ...)
            D2 = D2 < keep_prob  # 步骤2：将D2的值转换为0或1（使​​用keep_prob作为阈值）
            A2 = A2 * D2  # 步骤3：舍弃A1的一些节点（将它的值变为0或False）
            A2 = A2 / keep_prob  # 步骤4：缩放未舍弃的节点(不为0)的值

            Z3 = np.dot(W3, A2) + b3
            A3 =  sigmoid(Z3)

            cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

            return A3, cache

        def backward_propagation_with_dropout(X, Y, cache, keep_prob):
            """
            实现我们随机删除的模型的后向传播。
            参数：
                X  - 输入数据集，维度为（2，示例数）
                Y  - 标签，维度为（输出节点数量，示例数量）
                cache - 来自forward_propagation_with_dropout（）的cache输出
                keep_prob  - 随机删除的概率，实数

            返回：
                gradients - 一个关于每个参数、激活值和预激活变量的梯度值的字典
            """
            m = X.shape[1]
            (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

            dZ3 = A3 - Y
            dW3 = (1 / m) * np.dot(dZ3, A2.T)
            db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
            dA2 = np.dot(W3.T, dZ3)

            dA2 = dA2 * D2  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
            dA2 = dA2 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值

            dZ2 = np.multiply(dA2, np.int64(A2 > 0))
            dW2 = 1. / m * np.dot(dZ2, A1.T)
            db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.dot(W2.T, dZ2)

            dA1 = dA1 * D1  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
            dA1 = dA1 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值

            dZ1 = np.multiply(dA1, np.int64(A1 > 0))
            dW1 = 1. / m * np.dot(dZ1, X.T)
            db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

            gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                         "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                         "dZ1": dZ1, "dW1": dW1, "db1": db1}

            return gradients

        def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):
            """
            实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

            参数：
                X - 输入的数据，维度为(2, 要训练/测试的数量)
                Y - 标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
                learning_rate - 学习速率
                num_iterations - 迭代的次数
                print_cost - 是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
                is_polt - 是否绘制梯度下降的曲线图
                lambd - 正则化的超参数，实数
                keep_prob - 随机删除节点的概率
            返回
                parameters - 学习后的参数
            """
            grads = {}
            costs = []
            m = X.shape[1]
            layers_dims = [X.shape[0], 20, 3, 1]

            # 初始化参数
            parameters =  initialize_parameters(layers_dims)

            # 开始学习
            for i in range(0, num_iterations):
                # 前向传播
                ##是否随机删除节点
                if keep_prob == 1:
                    ###不随机删除节点
                    a3, cache =  forward_propagation(X, parameters)
                elif keep_prob < 1:
                    ###随机删除节点
                    a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
                else:
                    print("keep_prob参数错误！程序退出。")
                    exit

                # 计算成本
                ## 是否使用二范数
                if lambd == 0:
                    ###不使用L2正则化
                    cost =  compute_cost(a3, Y)
                else:
                    ###使用L2正则化
                    cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

                # 反向传播
                ##可以同时使用L2正则化和随机删除节点，但是本次实验不同时使用。
                assert (lambd == 0 or keep_prob == 1)

                ##两个参数的使用情况
                if (lambd == 0 and keep_prob == 1):
                    ### 不使用L2正则化和不使用随机删除节点
                    grads =  backward_propagation(X, Y, cache)
                elif lambd != 0:
                    ### 使用L2正则化，不使用随机删除节点
                    grads = backward_propagation_with_regularization(X, Y, cache, lambd)
                elif keep_prob < 1:
                    ### 使用随机删除节点，不使用L2正则化
                    grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

                # 更新参数
                parameters =  update_parameters(parameters, grads, learning_rate)

                # 记录并打印成本
                if i % 1000 == 0:
                    ## 记录成本
                    costs.append(cost)
                    if (print_cost and i % 10000 == 0):
                        # 打印成本
                        print("第" + str(i) + "次迭代，成本值为：" + str(cost))

            # 是否绘制成本曲线图
            if is_plot:
                plt.plot(costs)
                plt.ylabel('cost')
                plt.xlabel('iterations (x1,000)')
                plt.title("Learning rate =" + str(learning_rate))
                plt.show()

            # 返回学习后的参数
            return parameters


        train_X, train_Y, test_X, test_Y =  load_2D_dataset(is_plot=True)
        #不使用正则化
        #parameters = model(train_X, train_Y, is_plot=True)
        #L2正则化
        #parameters = model(train_X, train_Y, lambd=0.7, is_plot=True)
        #Dropout正则化
        parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3, is_plot=True)

        print("训练集:")
        predictions_train =  predict(train_X, train_Y, parameters)
        print("测试集:")
        predictions_test =  predict(test_X, test_Y, parameters)
        plt.title("Model without regularization")
        axes = plt.gca()
        axes.set_xlim([-0.75, 0.40])
        axes.set_ylim([-0.75, 0.65])
        plot_decision_boundary(lambda x:  predict_dec(parameters, x.T), train_X, train_Y)

    def gc_demo(self):
        """
        这个案例用来梯度检验
        Returns:
        """
        # -*- coding: utf-8 -*-

        import numpy as np
        import matplotlib.pyplot as plt

        def sigmoid(x):
            """
            Compute the sigmoid of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- sigmoid(x)
            """
            s = 1 / (1 + np.exp(-x))
            return s

        def relu(x):
            """
            Compute the relu of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- relu(x)
            """
            s = np.maximum(0, x)

            return s

        def dictionary_to_vector(parameters):
            """
            它将“参数”字典转换为称为“值”的向量，该向量是通过将所有参数(W1, b1, W2, b2, W3, b3)重塑为向量并将它们串联而获得的。

            反函数是“vector_to_dictionary”，它输出回“parameters”字典。
            Roll all our parameters dictionary into a single vector satisfying our specific required shape.
            """
            keys = []
            count = 0
            for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

                # flatten parameter
                new_vector = np.reshape(parameters[key], (-1, 1))
                keys = keys + [key] * new_vector.shape[0]

                if count == 0:
                    theta = new_vector
                else:
                    theta = np.concatenate((theta, new_vector), axis=0)
                count = count + 1

            return theta, keys

        def vector_to_dictionary(theta):
            """
            Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
            """
            parameters = {}
            parameters["W1"] = theta[:20].reshape((5, 4))
            parameters["b1"] = theta[20:25].reshape((5, 1))
            parameters["W2"] = theta[25:40].reshape((3, 5))
            parameters["b2"] = theta[40:43].reshape((3, 1))
            parameters["W3"] = theta[43:46].reshape((1, 3))
            parameters["b3"] = theta[46:47].reshape((1, 1))

            return parameters

        def gradients_to_vector(gradients):
            """
            Roll all our gradients dictionary into a single vector satisfying our specific required shape.
            """

            count = 0
            for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
                # flatten parameter
                new_vector = np.reshape(gradients[key], (-1, 1))

                if count == 0:
                    theta = new_vector
                else:
                    theta = np.concatenate((theta, new_vector), axis=0)
                count = count + 1

            return theta

        #一维的梯度检验
        def forward_propagation(x, theta):
            """

            实现图中呈现的线性前向传播（计算J）（J（theta）= theta * x）

            参数：
            x  - 一个实值输入
            theta  - 参数，也是一个实数

            返回：
            J  - 函数J的值，用公式J（theta）= theta * x计算
            """
            J = np.dot(theta, x)

            return J

        def backward_propagation(x, theta):
            """
            计算J相对于θ的导数。

            参数：
                x  - 一个实值输入
                theta  - 参数，也是一个实数

            返回：
                dtheta  - 相对于θ的成本梯度
            """
            dtheta = x

            return dtheta

        def gradient_check(x, theta, epsilon=1e-7):
            """

            实现图中的反向传播。 用的就是导数的最初的基本定义

            参数：
                x  - 一个实值输入
                theta  - 参数，也是一个实数
                epsilon  - 使用公式（3）计算输入的微小偏移以计算近似梯度

            返回：
                近似梯度和后向传播梯度之间的差异
            """

            # 使用公式（3）的左侧计算gradapprox。
            thetaplus = theta + epsilon  # Step 1
            thetaminus = theta - epsilon  # Step 2
            J_plus = forward_propagation(x, thetaplus)  # Step 3
            J_minus = forward_propagation(x, thetaminus)  # Step 4
            gradapprox = (J_plus - J_minus) / (2 * epsilon)  # Step 5

            # 检查gradapprox是否足够接近backward_propagation（）的输出
            grad = backward_propagation(x, theta)
            #求范数 实际就是公式 公式就是在比较范数
            numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
            denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
            difference = numerator / denominator  # Step 3'

            if difference < 1e-7:
                print("梯度检查：梯度正常!")
            else:
                print("梯度检查：梯度超出阈值!")

            return difference

        # 测试gradient_check
        print("-----------------测试一维度gradient_check-----------------")
        x, theta = 2, 4
        difference = gradient_check(x, theta)
        print("difference = " + str(difference))

        #高纬度的梯度检验
        def forward_propagation_n(X, Y, parameters):
            """
            实现图中的前向传播（并计算成本）。

            参数：
                X - 训练集为m个例子
                Y -  m个示例的标签
                parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
                    W1  - 权重矩阵，维度为（5,4）
                    b1  - 偏向量，维度为（5,1）
                    W2  - 权重矩阵，维度为（3,5）
                    b2  - 偏向量，维度为（3,1）
                    W3  - 权重矩阵，维度为（1,3）
                    b3  - 偏向量，维度为（1,1）

            返回：
                cost - 成本函数（logistic）
            """
            m = X.shape[1]
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]

            # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
            Z1 = np.dot(W1, X) + b1
            A1 =  relu(Z1)

            Z2 = np.dot(W2, A1) + b2
            A2 =  relu(Z2)

            Z3 = np.dot(W3, A2) + b3
            A3 =  sigmoid(Z3)

            # 计算成本
            logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
            cost = (1 / m) * np.sum(logprobs)

            cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

            return cost, cache
        def backward_propagation_n(X, Y, cache):
            """
            实现图中所示的反向传播。

            参数：
                X - 输入数据点（输入节点数量，1）
                Y - 标签
                cache - 来自forward_propagation_n（）的cache输出

            返回：
                gradients - 一个字典，其中包含与每个参数、激活和激活前变量相关的成本梯度。
            """
            m = X.shape[1]
            (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

            dZ3 = A3 - Y
            dW3 = (1. / m) * np.dot(dZ3, A2.T)
            dW3 = 1. / m * np.dot(dZ3, A2.T)
            db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

            dA2 = np.dot(W3.T, dZ3)
            dZ2 = np.multiply(dA2, np.int64(A2 > 0))
            # dW2 = 1. / m * np.dot(dZ2, A1.T) * 2  # Should not multiply by 2
            dW2 = 1. / m * np.dot(dZ2, A1.T)
            db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.dot(W2.T, dZ2)
            dZ1 = np.multiply(dA1, np.int64(A1 > 0))
            dW1 = 1. / m * np.dot(dZ1, X.T)
            # db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True) # Should not multiply by 4
            db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

            gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                         "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                         "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

            return gradients
        def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
            """
            检查backward_propagation_n是否正确计算forward_propagation_n输出的成本梯度

            参数：
                parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
                grad_output_propagation_n的输出包含与参数相关的成本梯度。
                x  - 输入数据点，维度为（输入节点数量，1）
                y  - 标签
                epsilon  - 计算输入的微小偏移以计算近似梯度

            返回：
                difference - 近似梯度和后向传播梯度之间的差异
            """
            # 初始化参数
            parameters_values, keys =  dictionary_to_vector(parameters)  # keys用不到
            grad =  gradients_to_vector(gradients)
            num_parameters = parameters_values.shape[0]
            J_plus = np.zeros((num_parameters, 1))
            J_minus = np.zeros((num_parameters, 1))
            gradapprox = np.zeros((num_parameters, 1))

            # 计算gradapprox
            for i in range(num_parameters):
                # 计算J_plus [i]。输入：“parameters_values，epsilon”。输出=“J_plus [i]”
                thetaplus = np.copy(parameters_values)  # Step 1
                thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
                J_plus[i], cache = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))  # Step 3 ，cache用不到

                # 计算J_minus [i]。输入：“parameters_values，epsilon”。输出=“J_minus [i]”。
                thetaminus = np.copy(parameters_values)  # Step 1
                thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
                J_minus[i], cache = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))  # Step 3 ，cache用不到

                # 计算gradapprox[i]
                gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

            # 通过计算差异比较gradapprox和后向传播梯度。
            numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
            denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
            difference = numerator / denominator  # Step 3'

            if difference < 1e-7:
                print("梯度检查：梯度正常!")
            else:
                print("梯度检查：梯度超出阈值!")

            return difference
        def gradient_check_n_test_case():
            pass
        X, Y, parameters = gradient_check_n_test_case()

        cost, cache = forward_propagation_n(X, Y, parameters)
        gradients = backward_propagation_n(X, Y, cache)
        difference = gradient_check_n(parameters, gradients, X, Y)

    def opt_demo(self):
        """
        此案例用来学习优化梯度下降的方法
       不使用任何优化算法
        mini-batch梯度下降法
        使用具有动量的梯度下降算法
         使用Adam算法
        Returns:
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.io
        import math
        import sklearn
        import sklearn.datasets
        # -*- coding: utf-8 -*-

        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn
        import sklearn.datasets

        def sigmoid(x):
            """
            Compute the sigmoid of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- sigmoid(x)
            """
            s = 1 / (1 + np.exp(-x))
            return s

        def relu(x):
            """
            Compute the relu of x

            Arguments:
            x -- A scalar or numpy array of any size.

            Return:
            s -- relu(x)
            """
            s = np.maximum(0, x)

            return s

        def load_params_and_grads(seed=1):
            np.random.seed(seed)
            W1 = np.random.randn(2, 3)
            b1 = np.random.randn(2, 1)
            W2 = np.random.randn(3, 3)
            b2 = np.random.randn(3, 1)

            dW1 = np.random.randn(2, 3)
            db1 = np.random.randn(2, 1)
            dW2 = np.random.randn(3, 3)
            db2 = np.random.randn(3, 1)

            return W1, b1, W2, b2, dW1, db1, dW2, db2

        def initialize_parameters(layer_dims):
            """
            Arguments:
            layer_dims -- python array (list) containing the dimensions of each layer in our network

            Returns:
            parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                            W1 -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                            b1 -- bias vector of shape (layer_dims[l], 1)
                            Wl -- weight matrix of shape (layer_dims[l-1], layer_dims[l])
                            bl -- bias vector of shape (1, layer_dims[l])

            Tips:
            - For example: the layer_dims for the "Planar Data classification model" would have been [2,2,1].
            This means W1's shape was (2,2), b1 was (1,2), W2 was (2,1) and b2 was (1,1). Now you have to generalize it!
            - In the for loop, use parameters['W' + str(l)] to access Wl, where l is the iterative integer.
            """

            np.random.seed(3)
            parameters = {}
            L = len(layer_dims)  # number of layers in the network

            for l in range(1, L):
                parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(
                    2 / layer_dims[l - 1])
                parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

                #assert (parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l - 1])
                #assert (parameters['W' + str(l)].shape == layer_dims[l], 1)

            return parameters

        def forward_propagation(X, parameters):
            """
            Implements the forward propagation (and computes the loss) presented in Figure 2.

            Arguments:
            X -- input dataset, of shape (input size, number of examples)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                            W1 -- weight matrix of shape ()
                            b1 -- bias vector of shape ()
                            W2 -- weight matrix of shape ()
                            b2 -- bias vector of shape ()
                            W3 -- weight matrix of shape ()
                            b3 -- bias vector of shape ()

            Returns:
            loss -- the loss function (vanilla logistic loss)
            """

            # retrieve parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]
            W3 = parameters["W3"]
            b3 = parameters["b3"]

            # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
            z1 = np.dot(W1, X) + b1
            a1 = relu(z1)
            z2 = np.dot(W2, a1) + b2
            a2 = relu(z2)
            z3 = np.dot(W3, a2) + b3
            a3 = sigmoid(z3)

            cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

            return a3, cache

        def backward_propagation(X, Y, cache):
            """
            Implement the backward propagation presented in figure 2.

            Arguments:
            X -- input dataset, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
            cache -- cache output from forward_propagation()

            Returns:
            gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
            """
            m = X.shape[1]
            (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

            dz3 = 1. / m * (a3 - Y)
            dW3 = np.dot(dz3, a2.T)
            db3 = np.sum(dz3, axis=1, keepdims=True)

            da2 = np.dot(W3.T, dz3)
            dz2 = np.multiply(da2, np.int64(a2 > 0))
            dW2 = np.dot(dz2, a1.T)
            db2 = np.sum(dz2, axis=1, keepdims=True)

            da1 = np.dot(W2.T, dz2)
            dz1 = np.multiply(da1, np.int64(a1 > 0))
            dW1 = np.dot(dz1, X.T)
            db1 = np.sum(dz1, axis=1, keepdims=True)

            gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                         "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                         "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}

            return gradients

        def compute_cost(a3, Y):

            """
            Implement the cost function

            Arguments:
            a3 -- post-activation, output of forward propagation
            Y -- "true" labels vector, same shape as a3

            Returns:
            cost - value of the cost function
            """
            m = Y.shape[1]

            logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
            cost = 1. / m * np.sum(logprobs)

            return cost

        def predict(X, y, parameters):
            """
            This function is used to predict the results of a  n-layer neural network.

            Arguments:
            X -- data set of examples you would like to label
            parameters -- parameters of the trained model

            Returns:
            p -- predictions for the given dataset X
            """

            m = X.shape[1]
            p = np.zeros((1, m) )

            # Forward propagation
            a3, caches = forward_propagation(X, parameters)

            # convert probas to 0/1 predictions
            for i in range(0, a3.shape[1]):
                if a3[0, i] > 0.5:
                    p[0, i] = 1
                else:
                    p[0, i] = 0

            # print results

            # print ("predictions: " + str(p[0,:]))
            # print ("true labels: " + str(y[0,:]))
            print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))

            return p

        def predict_dec(parameters, X):
            """
            Used for plotting decision boundary.

            Arguments:
            parameters -- python dictionary containing your parameters
            X -- input data of size (m, K)

            Returns
            predictions -- vector of predictions of our model (red: 0 / blue: 1)
            """

            # Predict using forward propagation and a classification threshold of 0.5
            a3, cache = forward_propagation(X, parameters)
            predictions = (a3 > 0.5)
            return predictions

        def plot_decision_boundary(model, X, y):
            # Set min and max values and give it some padding
            x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
            y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
            h = 0.01
            # Generate a grid of points with distance h between them
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            # Predict the function value for the whole grid
            Z = model(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            # Plot the contour and training examples
            plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
            plt.ylabel('x2')
            plt.xlabel('x1')
            plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
            plt.show()

        def load_dataset(is_plot=True):
            np.random.seed(3)
            train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2)  # 300 #0.2
            # Visualize the data
            if is_plot:
                plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
                plt.show()
            train_X = train_X.T
            train_Y = train_Y.reshape((1, train_Y.shape[0]))

            return train_X, train_Y

        def update_parameters_with_gd(parameters, grads, learning_rate):
            """
            使用梯度下降更新参数  minibatch-
            最基础的 也是我们以前一直在用的
            参数：
                parameters - 字典，包含了要更新的参数：
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
                grads - 字典，包含了每一个梯度值用以更新参数
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
                learning_rate - 学习率

            返回值：
                parameters - 字典，包含了更新后的参数
            """

            L = len(parameters) // 2  # 神经网络的层数

            # 更新每个参数
            for l in range(L):
                parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
                parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

            return parameters

        def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
            """
            从（X，Y）中创建一个随机的mini-batch列表

            参数：
                X - 输入数据，维度为(输入节点数量，样本的数量)
                Y - 对应的是X的标签，【1 | 0】（蓝|红），维度为(1,样本的数量)
                mini_batch_size - 每个mini-batch的样本数量

            返回：
                mini-bacthes - 一个同步列表，维度为（mini_batch_X,mini_batch_Y）

            """

            np.random.seed(seed)  # 指定随机种子
            m = X.shape[1]
            mini_batches = []

            # 第一步：打乱顺序
            permutation = list(np.random.permutation(m))  # 它会返回一个长度为m的随机数组，且里面的数是0到m-1
            shuffled_X = X[:, permutation]  # 将每一列的数据按permutation的顺序来重新排列。
            shuffled_Y = Y[:, permutation].reshape((1, m))

            """
            #博主注：
            #如果你不好理解的话请看一下下面的伪代码，看看X和Y是如何根据permutation来打乱顺序的。
            x = np.array([[1,2,3,4,5,6,7,8,9],
        				  [9,8,7,6,5,4,3,2,1]])
            y = np.array([[1,0,1,0,1,0,1,0,1]])

            random_mini_batches(x,y)
            permutation= [7, 2, 1, 4, 8, 6, 3, 0, 5]
            shuffled_X= [[8 3 2 5 9 7 4 1 6]
                         [2 7 8 5 1 3 6 9 4]]
            shuffled_Y= [[0 1 0 1 1 1 0 1 0]]
            """

            # 第二步，分割
            num_complete_minibatches = math.floor(
                m / mini_batch_size)  # 把你的训练集分割成多少份,请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
            for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
                mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
                """
                #博主注：
                #如果你不好理解的话请单独执行下面的代码，它可以帮你理解一些。
                a = np.array([[1,2,3,4,5,6,7,8,9],
                              [9,8,7,6,5,4,3,2,1],
                              [1,2,3,4,5,6,7,8,9]])
                k=1
                mini_batch_size=3
                print(a[:,1*3:(1+1)*3]) #从第4列到第6列
                '''
                [[4 5 6]
                 [6 5 4]
                 [4 5 6]]
                '''
                k=2
                print(a[:,2*3:(2+1)*3]) #从第7列到第9列
                '''
                [[7 8 9]
                 [3 2 1]
                 [7 8 9]]
                '''

                #看一下每一列的数据你可能就会好理解一些
                """
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            # 如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
            # 如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，我们要把它处理了
            if m % mini_batch_size != 0:
                # 获取最后剩余的部分
                mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_minibatches:]
                mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_minibatches:]

                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            return mini_batches

        def initialize_velocity(parameters):
            """
            初始化速度，velocity是一个字典：
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values:与相应的梯度/参数维度相同的值为零的矩阵。
            参数：
                parameters - 一个字典，包含了以下参数：
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
            返回:
                v - 一个字典变量，包含了以下参数：
                    v["dW" + str(l)] = dWl的速度
                    v["db" + str(l)] = dbl的速度

            """
            L = len(parameters) // 2  # 神经网络的层数
            v = {}

            for l in range(L):
                v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
                v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

            return v

        def update_parameters_with_momentun(parameters, grads, v, beta, learning_rate):
            """
            使用动量更新参数
            参数：
                parameters - 一个字典类型的变量，包含了以下字段：
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
                grads - 一个包含梯度值的字典变量，具有以下字段：
                    grads["dW" + str(l)] = dWl
                    grads["db" + str(l)] = dbl
                v - 包含当前速度的字典变量，具有以下字段：
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
                beta - 超参数，动量，实数
                learning_rate - 学习率，实数
            返回：
                parameters - 更新后的参数字典
                v - 包含了更新后的速度变量
            """
            L = len(parameters) // 2
            for l in range(L):
                # 计算速度
                v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
                v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

                # 更新参数
                parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * v["dW" + str(l + 1)]
                parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v["db" + str(l + 1)]

            return parameters, v

        def initialize_adam(parameters):
            """
            初始化v和s，它们都是字典类型的变量，都包含了以下字段：
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values：与对应的梯度/参数相同维度的值为零的numpy矩阵

            参数：
                parameters - 包含了以下参数的字典变量：
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
            返回：
                v - 包含梯度的指数加权平均值，字段如下：
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
                s - 包含平方梯度的指数加权平均值，字段如下：
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

            """

            L = len(parameters) // 2
            v = {}
            s = {}

            for l in range(L):
                v["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
                v["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

                s["dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
                s["db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])

            return (v, s)

        def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
            """
            使用Adam更新参数

            参数：
                parameters - 包含了以下字段的字典：
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
                grads - 包含了梯度值的字典，有以下key值：
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
                v - Adam的变量，第一个梯度的移动平均值，是一个字典类型的变量
                s - Adam的变量，平方梯度的移动平均值，是一个字典类型的变量
                t - 当前迭代的次数
                learning_rate - 学习率
                beta1 - 动量，超参数,用于第一阶段，使得曲线的Y值不从0开始（参见天气数据的那个图）
                beta2 - RMSprop的一个参数，超参数
                epsilon - 防止除零操作（分母为0）

            返回：
                parameters - 更新后的参数
                v - 第一个梯度的移动平均值，是一个字典类型的变量
                s - 平方梯度的移动平均值，是一个字典类型的变量
            """
            L = len(parameters) // 2
            v_corrected = {}  # 偏差修正后的值
            s_corrected = {}  # 偏差修正后的值

            for l in range(L):
                # 梯度的移动平均值,输入："v , grads , beta1",输出：" v "
                v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
                v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

                # 计算第一阶段的偏差修正后的估计值，输入"v , beta1 , t" , 输出："v_corrected"
                v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
                v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))

                # 计算平方梯度的移动平均值，输入："s, grads , beta2"，输出："s"
                s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
                s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])

                # 计算第二阶段的偏差修正后的估计值，输入："s , beta2 , t"，输出："s_corrected"
                s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2, t))
                s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

                # 更新参数，输入: "parameters, learning_rate, v_corrected, s_corrected, epsilon". 输出: "parameters".
                parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
                            v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
                parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (
                            v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))

            return (parameters, v, s)

        def model(X, Y, layers_dims, optimizer, learning_rate=0.0007,
                  mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
                  epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):

            """
            可以运行在不同优化器模式下的3层神经网络模型。

            参数：
                X - 输入数据，维度为（2，输入的数据集里面样本数量）
                Y - 与X对应的标签
                layers_dims - 包含层数和节点数量的列表
                optimizer - 字符串类型的参数，用于选择优化类型，【 "gd" | "momentum" | "adam" 】
                learning_rate - 学习率
                mini_batch_size - 每个小批量数据集的大小
                beta - 用于动量优化的一个超参数
                beta1 - 用于计算梯度后的指数衰减的估计的超参数
                beta1 - 用于计算平方梯度后的指数衰减的估计的超参数
                epsilon - 用于在Adam中避免除零操作的超参数，一般不更改
                num_epochs - 整个训练集的遍历次数，（视频2.9学习率衰减，1分55秒处，视频中称作“代”）,相当于之前的num_iteration
                print_cost - 是否打印误差值，每遍历1000次数据集打印一次，但是每100次记录一个误差值，又称每1000代打印一次
                is_plot - 是否绘制出曲线图

            返回：
                parameters - 包含了学习后的参数

            """
            L = len(layers_dims)
            costs = []
            t = 0  # 每学习完一个minibatch就增加1
            seed = 10  # 随机种子

            # 初始化参数
            parameters =  initialize_parameters(layers_dims)

            # 选择优化器
            if optimizer == "gd":
                pass  # 不使用任何优化器，直接使用梯度下降法
            elif optimizer == "momentum":
                v = initialize_velocity(parameters)  # 使用动量
            elif optimizer == "adam":
                v, s = initialize_adam(parameters)  # 使用Adam优化
            else:
                print("optimizer参数错误，程序退出。")
                exit(1)

            # 开始学习
            for i in range(num_epochs):
                # 定义随机 minibatches,我们在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
                seed = seed + 1
                minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

                for minibatch in minibatches:
                    # 选择一个minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # 前向传播
                    A3, cache =  forward_propagation(minibatch_X, parameters)

                    # 计算误差
                    cost =  compute_cost(A3, minibatch_Y)

                    # 反向传播
                    grads =  backward_propagation(minibatch_X, minibatch_Y, cache)

                    # 更新参数
                    if optimizer == "gd":
                        parameters = update_parameters_with_gd(parameters, grads, learning_rate)
                    elif optimizer == "momentum":
                        parameters, v = update_parameters_with_momentun(parameters, grads, v, beta, learning_rate)
                    elif optimizer == "adam":
                        t = t + 1
                        parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1,
                                                                       beta2, epsilon)
                # 记录误差值
                if i % 100 == 0:
                    costs.append(cost)
                    # 是否打印误差值
                    if print_cost and i % 1000 == 0:
                        print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
            # 是否绘制曲线图
            if is_plot:
                plt.plot(costs)
                plt.ylabel('cost')
                plt.xlabel('epochs (per 100)')
                plt.title("Learning rate = " + str(learning_rate))
                plt.show()

            return parameters


        train_X, train_Y =  load_dataset(is_plot=True)
        # 使用普通的梯度下降
        layers_dims = [train_X.shape[0], 5, 2, 1]
        parameters = model(train_X, train_Y, layers_dims, optimizer="gd", is_plot=True)

        # 使用动量的梯度下降
        #parameters = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum", is_plot=True)

        # 使用Adam优化的梯度下降
        #parameters = model(train_X, train_Y, layers_dims, optimizer="adam", is_plot=True)

        # 预测
        preditions =  predict(train_X, train_Y, parameters)

        # 绘制分类图
        plt.title("Model with  optimization")
        axes = plt.gca()
        axes.set_xlim([-1.5, 2.5])
        axes.set_ylim([-1, 1.5])
        plot_decision_boundary(lambda x:  predict_dec(parameters, x.T), train_X, train_Y)

    def tensorflow_demo(self):
        """
        此案例用于tensorflow的入门
        Returns:

        """
        import numpy as np
        import h5py
        import matplotlib.pyplot as plt
        import tensorflow.compat.v1 as tf #v1才有会话 v2就和常见的代码一样了
        tf.compat.v1.disable_eager_execution()
        import time

        np.random.seed(1)
        print("------------练习定义损失函数------------------")
        y_hat = tf.constant(36, name="y_hat")  # 定义y_hat为固定值36
        y = tf.constant(39, name="y")  # 定义y为固定值39
        # 利用feed_dict来改变x的值
        x = tf.placeholder(tf.int64, name="x")


        loss = tf.Variable((y - y_hat) ** 2, name="loss")  # 为损失函数创建一个变量

        init = tf.global_variables_initializer()  # 运行之后的初始化(ession.run(init))
        # 损失变量将被初始化并准备计算
        with tf.Session() as sess:  # 创建一个session并打印输出
            sess.run(init)  # 初始化变量
            print(sess.run(loss))  # 打印损失值
            print(sess.run(2 * x, feed_dict={x: 3}))

    def sign_language_demo(self):
        """
        此案例是通过手语识别的代码 来学习tensorflow
        Returns:

        """
        import h5py
        import numpy as np


        import tensorflow  as tf

        import math

        import matplotlib.pyplot as plt

        import time

        def load_dataset():
            """
            训练集：有从0到5的数字的1080张图片(64x64像素)，每个数字拥有180张图片。
            测试集：有从0到5的数字的120张图片(64x64像素)，每个数字拥有5张图片。
            Returns:

            """
            train_dataset = h5py.File('./datasets/L2W3/datasets/train_signs.h5', "r")
            train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
            train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

            test_dataset = h5py.File('./datasets/L2W3/datasets/test_signs.h5', "r")
            test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
            test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

            classes = np.array(test_dataset["list_classes"][:])  # the list of classes

            train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
            test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

            return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

        def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
            """
            Creates a list of random minibatches from (X, Y)

            Arguments:
            X -- input data, of shape (input size, number of examples)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
            mini_batch_size - size of the mini-batches, integer
            seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

            Returns:
            mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
            """

            m = X.shape[1]  # number of training examples
            mini_batches = []
            np.random.seed(seed)

            # Step 1: Shuffle (X, Y)
            permutation = list(np.random.permutation(m))
            shuffled_X = X[:, permutation]
            shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

            # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
            num_complete_minibatches = math.floor(
                m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
                mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            # Handling the end case (last mini-batch < mini_batch_size)
            if m % mini_batch_size != 0:
                mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
                mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            return mini_batches

        def convert_to_one_hot(Y, C):
            Y = np.eye(C)[Y.reshape(-1)].T
            return Y

        def predict(X, parameters):

            W1 = tf.convert_to_tensor(parameters["W1"])
            b1 = tf.convert_to_tensor(parameters["b1"])
            W2 = tf.convert_to_tensor(parameters["W2"])
            b2 = tf.convert_to_tensor(parameters["b2"])
            W3 = tf.convert_to_tensor(parameters["W3"])
            b3 = tf.convert_to_tensor(parameters["b3"])

            params = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}

            x = tf.placeholder("float", [12288, 1])

            z3 = forward_propagation_for_predict(x, params)
            p = tf.argmax(z3)

            sess = tf.Session()
            prediction = sess.run(p, feed_dict={x: X})

            return prediction

        def forward_propagation_for_predict(X, parameters):
            """
            Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

            Arguments:
            X -- input dataset placeholder, of shape (input size, number of examples)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                          the shapes are given in initialize_parameters
            Returns:
            Z3 -- the output of the last LINEAR unit
            """

            # Retrieve the parameters from the dictionary "parameters"
            W1 = parameters['W1']
            b1 = parameters['b1']
            W2 = parameters['W2']
            b2 = parameters['b2']
            W3 = parameters['W3']
            b3 = parameters['b3']
            # Numpy Equivalents:
            Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
            A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
            Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
            A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
            Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

            return Z3

        def create_placeholders(n_x, n_y):
            """
            为TensorFlow会话创建占位符
            参数：
                n_x - 一个实数，图片向量的大小（64*64*3 = 12288）
                n_y - 一个实数，分类数（从0到5，所以n_y = 6）

            返回：
                X - 一个数据输入的占位符，维度为[n_x, None]，dtype = "float"
                Y - 一个对应输入的标签的占位符，维度为[n_Y,None]，dtype = "float"

            提示：
                使用None，因为它让我们可以灵活处理占位符提供的样本数量。事实上，测试/训练期间的样本数量是不同的。

            """

            X = tf.placeholder(tf.float32, [n_x, None], name="X")
            Y = tf.placeholder(tf.float32, [n_y, None], name="Y")

            return X, Y

        def initialize_parameters():
            """
            初始化神经网络的参数，参数的维度如下：
                W1 : [25, 12288]
                b1 : [25, 1]
                W2 : [12, 25]
                b2 : [12, 1]
                W3 : [6, 12]
                b3 : [6, 1]

            返回：
                parameters - 包含了W和b的字典


            """

            tf.set_random_seed(1)  # 指定随机种子

            W1 = tf.get_variable("W1", [25, 12288], initializer=tf.initializers.glorot_uniform())
            b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
            W2 = tf.get_variable("W2", [12, 25], initializer=tf.initializers.glorot_uniform())
            b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
            W3 = tf.get_variable("W3", [6, 12], initializer=tf.initializers.glorot_uniform())
            b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

            parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2,
                          "W3": W3,
                          "b3": b3}

            return parameters

        def forward_propagation(X, parameters):
            """
            实现一个模型的前向传播，模型结构为LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

            参数：
                X - 输入数据的占位符，维度为（输入节点数量，样本数量）
                parameters - 包含了W和b的参数的字典

            返回：
                Z3 - 最后一个LINEAR节点的输出

            """

            W1 = parameters['W1']
            b1 = parameters['b1']
            W2 = parameters['W2']
            b2 = parameters['b2']
            W3 = parameters['W3']
            b3 = parameters['b3']

            Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
            # Z1 = tf.matmul(W1,X) + b1             #也可以这样写
            A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
            Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
            A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
            Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

            return Z3

        def compute_cost(Z3, Y):
            """
            计算成本

            参数：
                Z3 - 前向传播的结果
                Y - 标签，一个占位符，和Z3的维度相同

            返回：
                cost - 成本值


            """
            logits = tf.transpose(Z3)  # 转置
            labels = tf.transpose(Y)  # 转置

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

            return cost

        def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=1500, minibatch_size=32, print_cost=True, is_plot=True):
            """
            实现一个三层的TensorFlow神经网络：LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX

            参数：
                X_train - 训练集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 1080）
                Y_train - 训练集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 1080）
                X_test - 测试集，维度为（输入大小（输入节点数量） = 12288, 样本数量 = 120）
                Y_test - 测试集分类数量，维度为（输出大小(输出节点数量) = 6, 样本数量 = 120）
                learning_rate - 学习速率
                num_epochs - 整个训练集的遍历次数
                mini_batch_size - 每个小批量数据集的大小
                print_cost - 是否打印成本，每100代打印一次
                is_plot - 是否绘制曲线图

            返回：
                parameters - 学习后的参数

            """
            tf.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
            tf.set_random_seed(1)
            seed = 3
            (n_x, m) = X_train.shape  # 获取输入节点数量和样本数
            n_y = Y_train.shape[0]  # 获取输出节点数量
            costs = []  # 成本集

            # 给X和Y创建placeholder
            X, Y = create_placeholders(n_x, n_y)

            # 初始化参数
            parameters = initialize_parameters()

            # 前向传播
            Z3 = forward_propagation(X, parameters)

            # 计算成本
            cost = compute_cost(Z3, Y)

            # 反向传播，使用Adam优化
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            # 初始化所有的变量
            init = tf.global_variables_initializer()

            # 开始会话并计算
            with tf.Session() as sess:
                # 初始化
                sess.run(init)

                # 正常训练的循环
                for epoch in range(num_epochs):

                    epoch_cost = 0  # 每代的成本
                    num_minibatches = int(m / minibatch_size)  # minibatch的总数量
                    seed = seed + 1
                    minibatches =  random_mini_batches(X_train, Y_train, minibatch_size, seed)

                    for minibatch in minibatches:
                        # 选择一个minibatch
                        (minibatch_X, minibatch_Y) = minibatch

                        # 数据已经准备好了，开始运行session
                        _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                        # 计算这个minibatch在这一代中所占的误差
                        epoch_cost = epoch_cost + minibatch_cost / num_minibatches

                    # 记录并打印成本
                    ## 记录成本
                    if epoch % 5 == 0:
                        costs.append(epoch_cost)
                        # 是否打印：
                        if print_cost and epoch % 100 == 0:
                            print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))

                # 是否绘制图谱
                if is_plot:
                    plt.plot(np.squeeze(costs))
                    plt.ylabel('cost')
                    plt.xlabel('iterations (per tens)')
                    plt.title("Learning rate =" + str(learning_rate))
                    plt.show()

                # 保存学习后的参数
                parameters = sess.run(parameters)
                print("参数已经保存到session。")

                # 计算当前的预测结果
                correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

                # 计算准确率 tf.cast格式转换
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                """
                eval() 其实就是tf.Tensor的Session.run() 的另外一种写法，但两者有差别
                eval(): 将字符串string对象转化为有效的表达式参与求值运算返回计算结果
                eval()也是启动计算的一种方式。基于Tensorflow的基本原理，首先需要定义图，然后计算图，其中计算图的函数常见的有run()函数，如sess.run()。同样eval()也是此类函数，
                要注意的是，eval()只能用于tf.Tensor类对象，也就是有输出的Operation。对于没有输出的Operation, 可以用.run()或者Session.run()；Session.run()没有这个限制。
                """
                print("训练集的准确率：", accuracy.eval({X: X_train, Y: Y_train}))
                print("测试集的准确率:", accuracy.eval({X: X_test, Y: Y_test}))

                return parameters


        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes =  load_dataset()
        index = 11
        plt.imshow(X_train_orig[index])
        plt.show()
        print("Y = " + str(np.squeeze(Y_train_orig[:, index])))

        X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T  # 每一列就是一个样本
        X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

        # 归一化数据
        X_train = X_train_flatten / 255
        X_test = X_test_flatten / 255

        # 转换为独热矩阵
        Y_train =  convert_to_one_hot(Y_train_orig, 6)
        Y_test =  convert_to_one_hot(Y_test_orig, 6)

        print("训练集样本数 = " + str(X_train.shape[1]))
        print("测试集样本数 = " + str(X_test.shape[1]))
        print("X_train.shape: " + str(X_train.shape))
        print("Y_train.shape: " + str(Y_train.shape))
        print("X_test.shape: " + str(X_test.shape))
        print("Y_test.shape: " + str(Y_test.shape))

        # 开始时间
        start_time = time.perf_counter()
        # 开始训练
        parameters = model(X_train, Y_train, X_test, Y_test)
        # 结束时间
        end_time = time.perf_counter()
        # 计算时差
        print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")

    def cnn_demo(self):
        """
        此案例实现卷积神经网络的基本机构  还是手语识别的案例
        Returns:

        """
        import math
        import numpy as np
        import h5py
        import matplotlib.pyplot as plt
        import tensorflow as tf
        from tensorflow.python.framework import ops

        #---------------------手动实现---------------------
        def zero_pad(X, pad)->np.ndarray:
            """
            把数据集X的图像边界全部使用0来扩充pad个宽度和高度。

            参数：
                X - 图像数据集，维度为（样本数，图像高度，图像宽度，图像通道数）
                pad - 整数，每个图像在垂直和水平维度上的填充量
            返回：
                X_paded - 扩充后的图像数据集，维度为（样本数，图像高度 + 2*pad，图像宽度 + 2*pad，图像通道数）

            """

            X_paded = np.pad(X, (
                (0, 0),  # 样本数，不填充
                (pad, pad),  # 图像高度,你可以视为上面填充x个，下面填充y个(x,y)
                (pad, pad),  # 图像宽度,你可以视为左边填充x个，右边填充y个(x,y)
                (0, 0)),  # 通道数，不填充
                             'constant', constant_values=0)  # 连续一样的值填充

            return X_paded

        def conv_single_step(a_slice_prev, W, b):
            """
            在前一层的激活输出的一个片段上应用一个由参数W定义的过滤器。
            这里切片大小和过滤器大小相同

            参数：
                a_slice_prev - 输入数据的一个片段，维度为（过滤器大小，过滤器大小，上一通道数）
                W - 权重参数，包含在了一个矩阵中，维度为（过滤器大小，过滤器大小，上一通道数）
                b - 偏置参数，包含在了一个矩阵中，维度为（1,1,1）

            返回：
                Z - 在输入数据的片X上卷积滑动窗口（w，b）的结果。
            """

            s = np.multiply(a_slice_prev, W) + b

            Z = np.sum(s)

            return Z

        def conv_forward(A_prev, W, b, hparameters):
            """
            实现卷积函数的前向传播

            参数：
                A_prev - 上一层的激活输出矩阵，维度为(m, n_H_prev, n_W_prev, n_C_prev)，（样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量）
                W - 权重矩阵，维度为(f, f, n_C_prev, n_C)，（过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量）
                b - 偏置矩阵，维度为(1, 1, 1, n_C)，（1,1,1,这一层的过滤器数量）
                hparameters - 包含了"stride"与 "pad"的超参数字典。

            返回：
                Z - 卷积输出，维度为(m, n_H, n_W, n_C)，（样本数，图像的高度，图像的宽度，过滤器数量）
                cache - 缓存了一些反向传播函数conv_backward()需要的一些数据
            """

            # 获取来自上一层数据的基本信息  样本数 上一层的图像高度 宽度  n_c是滤波器数量 如果有5个滤波器 对于原始三通道图像就是 滤波器就是5*3
            (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape#样本数量，上一层图像的高度，上一层图像的宽度，上一层过滤器数量

            # 获取权重矩阵的基本信息
            (f, f, n_C_prev, n_C) = W.shape#过滤器大小，过滤器大小，上一层的过滤器数量，这一层的过滤器数量

            # 获取超参数hparameters的值
            stride = hparameters["stride"]
            pad = hparameters["pad"]

            # 计算卷积后的图像的宽度高度 通道数就是过滤器数量，参考上面的公式，使用int()来进行板除
            n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
            n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

            # 使用0来初始化卷积输出Z
            Z = np.zeros((m, n_H, n_W, n_C))

            # 通过A_prev创建填充过了的A_prev_pad
            A_prev_pad = zero_pad(A_prev, pad)

            for i in range(m):  # 遍历样本
                a_prev_pad = A_prev_pad[i]  # 选择第i个样本的扩充后的激活矩阵
                for h in range(n_H):  # 在输出的垂直轴上循环
                    for w in range(n_W):  # 在输出的水平轴上循环
                        for c in range(n_C):  # 循环遍历 每一个滤波器
                            # 定位当前的切片位置
                            vert_start = h * stride  # 竖向，开始的位置
                            vert_end = vert_start + f  # 竖向，结束的位置
                            horiz_start = w * stride  # 横向，开始的位置
                            horiz_end = horiz_start + f  # 横向，结束的位置
                            # 切片位置定位好了我们就把它取出来, 这里为什么是所有的通道数呢 因为上一层循环是遍历的滤波器数量 这和滤波器的通道数没有关系
                            a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                            # 执行单步卷积 这里的:, :, : 代表这个c的滤波器的所有高度 宽度 通道数

                            Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[0, 0, 0, c])

            # 数据处理完毕，验证数据格式是否正确
            assert (Z.shape == (m, n_H, n_W, n_C))

            # 存储一些缓存值，以便于反向传播使用
            cache = (A_prev, W, b, hparameters)

            return (Z, cache)

        def conv_backward(dZ, cache):
            """
            实现卷积层的反向传播

            参数：
                dZ - 卷积层的输出Z的 梯度，维度为(m, n_H, n_W, n_C)
                cache - 反向传播所需要的参数，conv_forward()的输出之一

            返回：
                dA_prev - 卷积层的输入（A_prev）的梯度值，维度为(m, n_H_prev, n_W_prev, n_C_prev)
                dW - 卷积层的权值的梯度，维度为(f,f,n_C_prev,n_C)
                db - 卷积层的偏置的梯度，维度为（1,1,1,n_C）

            """
            # 获取cache的值
            (A_prev, W, b, hparameters) = cache

            # 获取A_prev的基本信息
            (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

            # 获取dZ的基本信息
            (m, n_H, n_W, n_C) = dZ.shape

            # 获取权值的基本信息
            (f, f, n_C_prev, n_C) = W.shape

            # 获取hparaeters的值
            pad = hparameters["pad"]
            stride = hparameters["stride"]

            # 初始化各个梯度的结构 不止要dw db 这是用于更新的梯度 dA_prev是卷积层输入的梯度 作为下一层需要计算梯度时的输入就好像这一层的dz
            dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
            dW = np.zeros((f, f, n_C_prev, n_C))
            db = np.zeros((1, 1, 1, n_C))

            # 前向传播中我们使用了pad，反向传播也需要使用，这是为了保证数据结构一致
            A_prev_pad = zero_pad(A_prev, pad)
            dA_prev_pad = zero_pad(dA_prev, pad)

            # 现在处理数据
            for i in range(m):
                # 选择第i个扩充了的数据的样本,降了一维。
                a_prev_pad = A_prev_pad[i]
                da_prev_pad = dA_prev_pad[i]

                for h in range(n_H):
                    for w in range(n_W):
                        for c in range(n_C):
                            # 定位切片位置
                            vert_start = h
                            vert_end = vert_start + f
                            horiz_start = w
                            horiz_end = horiz_start + f

                            # 定位完毕，开始切片
                            a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                            # 切片完毕，使用上面的公式计算梯度
                            da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                            dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                            db[:, :, :, c] += dZ[i, h, w, c]
                # 设置第i个样本最终的dA_prev,即把非填充的数据取出来。
                dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

            # 数据处理完毕，验证数据格式是否正确
            assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

            return (dA_prev, dW, db)

        def pool_forward(A_prev, hparameters, mode="max"):
            """
            实现池化层的前向传播

            参数：
                A_prev - 输入数据，维度为(m, n_H_prev, n_W_prev, n_C_prev)
                hparameters - 包含了 "f" 和 "stride"的超参数字典
                mode - 模式选择【"max" | "average"】

            返回：
                A - 池化层的输出，维度为 (m, n_H, n_W, n_C)
                cache - 存储了一些反向传播需要用到的值，包含了输入和超参数的字典。
            """

            # 获取输入数据的基本信息
            (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

            # 获取超参数的信息
            f = hparameters["f"]
            stride = hparameters["stride"]

            # 计算输出维度
            n_H = int((n_H_prev - f) / stride) + 1
            n_W = int((n_W_prev - f) / stride) + 1
            n_C = n_C_prev

            # 初始化输出矩阵
            A = np.zeros((m, n_H, n_W, n_C))

            for i in range(m):  # 遍历样本
                for h in range(n_H):  # 在输出的垂直轴上循环
                    for w in range(n_W):  # 在输出的水平轴上循环
                        for c in range(n_C):  # 循环遍历输出的通道
                            # 定位当前的切片位置
                            vert_start = h * stride  # 竖向，开始的位置
                            vert_end = vert_start + f  # 竖向，结束的位置
                            horiz_start = w * stride  # 横向，开始的位置
                            horiz_end = horiz_start + f  # 横向，结束的位置
                            # 定位完毕，开始切割
                            a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                            # 对切片进行池化操作
                            if mode == "max":
                                A[i, h, w, c] = np.max(a_slice_prev)
                            elif mode == "average":
                                A[i, h, w, c] = np.mean(a_slice_prev)

            # 池化完毕，校验数据格式
            assert (A.shape == (m, n_H, n_W, n_C))

            # 校验完毕，开始存储用于反向传播的值
            cache = (A_prev, hparameters)

            return A, cache

        def create_mask_from_window(x):
            """
            从输入矩阵中创建掩码，以保存最大值的矩阵的位置。

            参数：
                x - 一个维度为(f,f)的矩阵

            返回：
                mask - 包含x的最大值的位置的矩阵
            """
            mask = x == np.max(x)

            return mask

        def distribute_value(dz, shape):
            """
            给定一个值，为按矩阵大小平均分配到每一个矩阵位置中。

            参数：
                dz - 输入的实数
                shape - 元组，两个值，分别为n_H , n_W

            返回：
                a - 已经分配好了值的矩阵，里面的值全部一样。

            """
            # 获取矩阵的大小
            (n_H, n_W) = shape

            # 计算平均值
            average = dz / (n_H * n_W)

            # 填充入矩阵
            a = np.ones(shape) * average

            return a

        def pool_backward(dA, cache, mode="max"):
            """
            实现池化层的反向传播

            参数:
                dA - 池化层的输出的梯度，和池化层的输出的维度一样
                cache - 池化层前向传播时所存储的参数。
                mode - 模式选择，【"max" | "average"】

            返回：
                dA_prev - 池化层的输入的梯度，和A_prev的维度相同

            """
            # 获取cache中的值
            (A_prev, hparaeters) = cache

            # 获取hparaeters的值
            f = hparaeters["f"]
            stride = hparaeters["stride"]

            # 获取A_prev和dA的基本信息
            (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
            (m, n_H, n_W, n_C) = dA.shape

            # 初始化输出的结构
            dA_prev = np.zeros_like(A_prev)

            # 开始处理数据
            for i in range(m):
                a_prev = A_prev[i]
                for h in range(n_H):
                    for w in range(n_W):
                        for c in range(n_C):
                            # 定位切片位置
                            vert_start = h
                            vert_end = vert_start + f
                            horiz_start = w
                            horiz_end = horiz_start + f

                            # 选择反向传播的计算方式
                            if mode == "max":
                                # 开始切片
                                a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                                # 创建掩码
                                mask = create_mask_from_window(a_prev_slice)
                                # 计算dA_prev
                                dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask,
                                                                                                         dA[i, h, w, c])

                            elif mode == "average":
                                # 获取dA的值
                                da = dA[i, h, w, c]
                                # 定义过滤器大小
                                shape = (f, f)
                                # 平均分配
                                dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
            # 数据处理完毕，开始验证格式
            assert (dA_prev.shape == A_prev.shape)

            return dA_prev

        #------------tensorflow的实现 以及案例---------------

        def load_dataset():
            """
            训练集：有从0到5的数字的1080张图片(64x64像素)，每个数字拥有180张图片。
            测试集：有从0到5的数字的120张图片(64x64像素)，每个数字拥有5张图片。
            Returns:

            """
            train_dataset = h5py.File('./datasets/L2W3/datasets/train_signs.h5', "r")
            train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
            train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

            test_dataset = h5py.File('./datasets/L2W3/datasets/test_signs.h5', "r")
            test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
            test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

            classes = np.array(test_dataset["list_classes"][:])  # the list of classes

            train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
            test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

            return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

        def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
            """
            Creates a list of random minibatches from (X, Y)
            Arguments:
            X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
            Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
            mini_batch_size - size of the mini-batches, integer
            seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
            Returns:
            mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
            """
            m = X.shape[0]  # number of training examples
            mini_batches = []
            np.random.seed(seed)
            # Step 1: Shuffle (X, Y)
            permutation = list(np.random.permutation(m))
            shuffled_X = X[permutation, :, :, :]
            shuffled_Y = Y[permutation, :]
            # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
            num_complete_minibatches = math.floor(
                m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
            for k in range(0, num_complete_minibatches):
                mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
                mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)
            # Handling the end case (last mini-batch &lt; mini_batch_size)
            if m % mini_batch_size != 0:
                mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
                mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)
            return mini_batches

        def convert_to_one_hot(Y, C):
            Y = np.eye(C)[Y.reshape(-1)].T
            return Y

        def forward_propagation_for_predict(X, parameters):
            """
            Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
            Arguments:
            X -- input dataset placeholder, of shape (input size, number of examples)
            parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                          the shapes are given in initialize_parameters
            Returns:
            Z3 -- the output of the last LINEAR unit
            """
            # Retrieve the parameters from the dictionary "parameters"
            W1 = parameters['W1']
            b1 = parameters['b1']
            W2 = parameters['W2']
            b2 = parameters['b2']
            W3 = parameters['W3']
            b3 = parameters['b3']
            # Numpy Equivalents:
            Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
            A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
            Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
            A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
            Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
            return Z3

        def predict(X, parameters):
            W1 = tf.convert_to_tensor(parameters["W1"])
            b1 = tf.convert_to_tensor(parameters["b1"])
            W2 = tf.convert_to_tensor(parameters["W2"])
            b2 = tf.convert_to_tensor(parameters["b2"])
            W3 = tf.convert_to_tensor(parameters["W3"])
            b3 = tf.convert_to_tensor(parameters["b3"])
            params = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
            x = tf.placeholder("float", [12288, 1])
            z3 = forward_propagation_for_predict(x, params)
            p = tf.argmax(z3)
            sess = tf.Session()
            prediction = sess.run(p, feed_dict={x: X})
            return prediction

        def create_placeholders(n_H0, n_W0, n_C0, n_y):
            """
            为session创建占位符

            参数：
                n_H0 - 实数，输入图像的高度
                n_W0 - 实数，输入图像的宽度
                n_C0 - 实数，输入的通道数
                n_y  - 实数，分类数

            输出：
                X - 输入数据的占位符，维度为[None, n_H0, n_W0, n_C0]，类型为"float"
                Y - 输入数据的标签的占位符，维度为[None, n_y]，维度为"float"
            """
            X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
            Y = tf.placeholder(tf.float32, [None, n_y])

            return X, Y

        def initialize_parameters():
            """
            初始化权值矩阵，这里我们把权值矩阵硬编码：
            W1 : [4, 4, 3, 8]
            W2 : [2, 2, 8, 16]

            返回：
                包含了tensor类型的W1、W2的字典
            """
            tf.set_random_seed(1)

            W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
            W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

            parameters = {"W1": W1,
                          "W2": W2}

            return parameters

        def forward_propagation(X, parameters):
            """
            实现前向传播
            CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

            参数：
                X - 输入数据的placeholder，维度为(输入节点数量，样本数量)
                parameters - 包含了“W1”和“W2”的python字典。

            返回：
                Z3 - 最后一个LINEAR节点的输出

            """
            W1 = parameters['W1']
            W2 = parameters['W2']

            # Conv2d : 步伐：1，填充方式：“SAME”
            Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
            # ReLU ：
            A1 = tf.nn.relu(Z1)
            # Max pool : 窗口大小：8x8，步伐：8x8，填充方式：“SAME”
            P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

            # Conv2d : 步伐：1，填充方式：“SAME”
            Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="SAME")
            # ReLU ：
            A2 = tf.nn.relu(Z2)
            # Max pool : 过滤器大小：4x4，步伐：4x4，填充方式：“SAME”
            P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

            # 一维化上一层的输出
            P = tf.contrib.layers.flatten(P2)

            # 全连接层（FC）：使用没有非线性激活函数的全连接层
            Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

            return Z3

        def compute_cost(Z3, Y):
            """
            计算成本
            参数：
                Z3 - 正向传播最后一个LINEAR节点的输出，维度为（6，样本数）。
                Y - 标签向量的placeholder，和Z3的维度相同

            返回：
                cost - 计算后的成本

            """

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

            return cost

        def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
                  num_epochs=100, minibatch_size=64, print_cost=True, isPlot=True):
            """
            使用TensorFlow实现三层的卷积神经网络
            CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
            number of training examples = 1080
            number of test examples = 120
            X_train shape: (1080, 64, 64, 3)
            Y_train shape: (1080, 6)
            X_test shape: (120, 64, 64, 3)
            Y_test shape: (120, 6)
            参数：
                X_train - 训练数据，维度为(None, 64, 64, 3)
                Y_train - 训练数据对应的标签，维度为(None, n_y = 6)
                X_test - 测试数据，维度为(None, 64, 64, 3)
                Y_test - 训练数据对应的标签，维度为(None, n_y = 6)
                learning_rate - 学习率
                num_epochs - 遍历整个数据集的次数
                minibatch_size - 每个小批量数据块的大小
                print_cost - 是否打印成本值，每遍历100次整个数据集打印一次
                isPlot - 是否绘制图谱

            返回：
                train_accuracy - 实数，训练集的准确度
                test_accuracy - 实数，测试集的准确度
                parameters - 学习后的参数
            """
            ops.reset_default_graph()  # 能够重新运行模型而不覆盖tf变量
            tf.set_random_seed(1)  # 确保你的数据和我一样
            seed = 3  # 指定numpy的随机种子
            #n_c n_c_pre 这是用于卷积时的参数 原始图像是n_c_pre 3通道 与三通道的算子相乘相加为一通道 n_c就是需要几个算子 一个算子合成一个通道
            (m, n_H0, n_W0, n_C0) = X_train.shape #  (1080, 64, 64, 3)
            n_y = Y_train.shape[1]  #(1080, 6)
            costs = []

            # 为当前维度创建占位符
            X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

            # 初始化参数
            parameters = initialize_parameters()

            # 前向传播
            Z3 = forward_propagation(X, parameters)

            # 计算成本
            cost = compute_cost(Z3, Y)

            # 反向传播，由于框架已经实现了反向传播，我们只需要选择一个优化器就行了
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            # 全局初始化所有变量
            init = tf.global_variables_initializer()

            # 开始运行
            with tf.Session() as sess:
                # 初始化参数
                sess.run(init)
                # 开始遍历数据集
                for epoch in range(num_epochs):
                    minibatch_cost = 0
                    num_minibatches = int(m / minibatch_size)  # 获取数据块的数量
                    seed = seed + 1
                    minibatches =  random_mini_batches(X_train, Y_train, minibatch_size, seed)

                    # 对每个数据块进行处理
                    for minibatch in minibatches:
                        # 选择一个数据块
                        (minibatch_X, minibatch_Y) = minibatch
                        # 最小化这个数据块的成本
                        _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                        # 累加数据块的成本值
                        minibatch_cost += temp_cost / num_minibatches

                    # 是否打印成本
                    if print_cost:
                        # 每5代打印一次
                        if epoch % 5 == 0:
                            print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

                    # 记录成本
                    if epoch % 1 == 0:
                        costs.append(minibatch_cost)

                # 数据处理完毕，绘制成本曲线
                if isPlot:
                    plt.plot(np.squeeze(costs))
                    plt.ylabel('cost')
                    plt.xlabel('iterations (per tens)')
                    plt.title("Learning rate =" + str(learning_rate))
                    plt.show()

                # 开始预测数据
                ## 计算当前的预测情况
                predict_op = tf.argmax(Z3, 1)
                corrent_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

                ##计算准确度
                accuracy = tf.reduce_mean(tf.cast(corrent_prediction, "float"))
                print("corrent_prediction accuracy= " + str(accuracy))

                train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
                test_accuary = accuracy.eval({X: X_test, Y: Y_test})

                print("训练集准确度：" + str(train_accuracy))
                print("测试集准确度：" + str(test_accuary))

                return (train_accuracy, test_accuary, parameters)

        import math
        import numpy as np
        import h5py
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import tensorflow as tf
        #from tensorflow.python.framework import ops

        np.random.seed(1)  # 指定随机种子

        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
        index = 0
        plt.imshow(X_train_orig[index])
        plt.show()
        print("Y = " + str(np.squeeze(Y_train_orig[:, index])))

        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.

        Y_train =  convert_to_one_hot(Y_train_orig, 6).T
        Y_test =  convert_to_one_hot(Y_test_orig, 6).T
        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))
        conv_layers = {}
        #CONV2D→RELU→MAXPOOL→CONV2D→RELU→MAXPOOL→FULLCONNECTED
        _, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=150)
    def smiling_face(self):
        """
        检测笑脸 练习使用keras
        Returns:

        """
        import numpy as np
        from tensorflow.keras import layers
        from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
        from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
        from tensorflow.keras.models import Model
        from tensorflow.keras.preprocessing import image
        from tensorflow.python.keras.utils import layer_utils
        from tensorflow.python.keras.utils.data_utils import get_file
        from tensorflow.keras.applications.imagenet_utils import preprocess_input
        import pydot
        from IPython.display import SVG
        from tensorflow.python.keras.utils.vis_utils import model_to_dot
        from tensorflow.keras.utils import plot_model
        import tensorflow.keras.backend as K
        K.set_image_data_format('channels_last')
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import imshow
        import math
        import h5py


        def mean_pred(y_true, y_pred):
            return K.mean(y_pred)

        def load_dataset():
            train_dataset = h5py.File('./datasets/L4W2/datasets/train_happy.h5', "r")
            train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
            train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

            test_dataset = h5py.File('./datasets/L4W2/datasets/test_happy.h5', "r")
            test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
            test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

            classes = np.array(test_dataset["list_classes"][:])  # the list of classes

            train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
            test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

            return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

        def HappyModel(input_shape):
            """
            实现一个检测笑容的模型

            参数：
                input_shape - 输入的数据的维度
            返回：
                model - 创建的Keras的模型

            """

            # 你可以参考和上面的大纲
            X_input = Input(input_shape)

            # 使用0填充：X_input的周围填充0
            X = ZeroPadding2D((3, 3))(X_input)

            # 对X使用 CONV -> BN -> RELU 块
            X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
            X = BatchNormalization(axis=3, name='bn0')(X)
            X = Activation('relu')(X)

            # 最大值池化层
            X = MaxPooling2D((2, 2), name='max_pool')(X)

            # 降维，矩阵转化为向量 + 全连接层
            X = Flatten()(X)
            X = Dense(1, activation='sigmoid', name='fc')(X)

            # 创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
            model = Model(inputs=X_input, outputs=X, name='HappyModel')

            return model


        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

        # Normalize image vectors
        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.

        # Reshape 输出为0或1 代表笑还是不笑 所以形状是（样本数，1） 如果输出是很多类 如6类 （样本数，6） 这6是6个概率
        Y_train = Y_train_orig.T
        Y_test = Y_test_orig.T

        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))

        # 创建一个模型实体
        happy_model = HappyModel(X_train.shape[1:])
        # 编译模型
        happy_model.compile("adam", "binary_crossentropy", metrics=['accuracy'])
        happy_model.summary()

        #plot_model(happy_model, to_file='happy_model.png')
        #SVG(model_to_dot(happy_model).create(prog='dot', format='svg'))

        # 训练模型
        # 请注意，此操作会花费你大约6-10分钟。
        happy_model.fit(X_train, Y_train, epochs=4, batch_size=50)
        # 评估模型
        preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
        print("误差值 = " + str(preds[0]))
        print("准确度 = " + str(preds[1]))

    def resnets_demo(self):
        """
        学习残差网络 实现resnets网络 用的是手写识别案例
        Returns:
        """

        import os
        import numpy as np
        import tensorflow as tf
        import h5py
        import math
        import numpy as np
        import tensorflow as tf

        from tensorflow.keras import layers
        from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
            AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
        from tensorflow.python.keras.models import Model, load_model
        from tensorflow.python.keras.preprocessing import image
        from tensorflow.python.keras.utils import layer_utils
        from tensorflow.python.keras.utils.data_utils import get_file
        from tensorflow.keras.applications.imagenet_utils import preprocess_input
        from tensorflow.python.keras.utils.vis_utils import model_to_dot
        from tensorflow.keras.utils import plot_model
        from tensorflow.keras.initializers import glorot_uniform

        import pydot
        from IPython.display import SVG
        import scipy.misc
        from matplotlib.pyplot import imshow
        import tensorflow.keras.backend as K
        K.set_image_data_format('channels_last')
        K.set_learning_phase(1)


        def load_dataset():
            train_dataset = h5py.File('./datasets/L4W2/datasets/train_signs.h5', "r")
            train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
            train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

            test_dataset = h5py.File('./datasets/L4W2/datasets/test_signs.h5', "r")
            test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
            test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

            classes = np.array(test_dataset["list_classes"][:])  # the list of classes

            train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
            test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

            return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

        def convert_to_one_hot(Y, C):
            Y = np.eye(C)[Y.reshape(-1)].T
            return Y
        def identity_block(X, f, filters, stage, block):
            """
            实现图3的恒等块

            参数：
                X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_H_prev )
                f - 整数，指定主路径中间的CONV窗口的维度
                filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
                stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
                block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。

            返回：
                X - 恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)

            """

            # 定义命名规则
            conv_name_base = "res" + str(stage) + block + "_branch"
            bn_name_base = "bn" + str(stage) + block + "_branch"

            # 获取过滤器
            F1, F2, F3 = filters

            # 保存输入数据，将会用于为主路径添加捷径
            X_shortcut = X

            # 主路径的第一部分
            ##卷积层
            X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                       name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
            ##归一化
            X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
            ##使用ReLU激活函数
            X = Activation("relu")(X)

            # 主路径的第二部分
            ##卷积层
            X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
                       name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
            ##归一化
            X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
            ##使用ReLU激活函数
            X = Activation("relu")(X)

            # 主路径的第三部分
            ##卷积层
            X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                       name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
            ##归一化
            X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
            ##没有ReLU激活函数

            # 最后一步：
            ##将捷径与输入加在一起
            X = Add()([X, X_shortcut])
            ##使用ReLU激活函数
            X = Activation("relu")(X)

            return X

        def convolutional_block(X, f, filters, stage, block, s=2):
            """
            实现图5的卷积块

            参数：
                X - 输入的tensor类型的变量，维度为( m, n_H_prev, n_W_prev, n_C_prev)
                f - 整数，指定主路径中间的CONV窗口的维度
                filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
                stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
                block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
                s - 整数，指定要使用的步幅

            返回：
                X - 卷积块的输出，tensor类型，维度为(n_H, n_W, n_C)
            """

            # 定义命名规则
            conv_name_base = "res" + str(stage) + block + "_branch"
            bn_name_base = "bn" + str(stage) + block + "_branch"

            # 获取过滤器数量
            F1, F2, F3 = filters

            # 保存输入数据
            X_shortcut = X

            # 主路径
            ##主路径第一部分
            X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid",
                       name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
            X = Activation("relu")(X)

            ##主路径第二部分
            X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
                       name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
            X = Activation("relu")(X)

            ##主路径第三部分
            X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
                       name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

            # 捷径
            X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="valid",
                                name=conv_name_base + "1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
            X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

            # 最后一步
            X = Add()([X, X_shortcut])
            X = Activation("relu")(X)

            return X

        def ResNet50(input_shape=(64, 64, 3), classes=6):
            """
            实现ResNet50
            CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
            -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

            参数：
                input_shape - 图像数据集的维度
                classes - 整数，分类数

            返回：
                model - Keras框架的模型

            """

            # 定义tensor类型的输入数据
            X_input = Input(input_shape)

            # 0填充
            X = ZeroPadding2D((3, 3))(X_input)

            # stage1
            X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv1",
                       kernel_initializer=glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis=3, name="bn_conv1")(X)
            X = Activation("relu")(X)
            X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

            # stage2
            X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
            X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
            X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")

            # stage3
            X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
            X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
            X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
            X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

            # stage4
            X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
            X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
            X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
            X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
            X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
            X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

            # stage5
            X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
            X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
            X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

            # 均值池化层
            X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

            # 输出层
            X = Flatten()(X)
            X = Dense(classes, activation="softmax", name="fc" + str(classes),
                      kernel_initializer=glorot_uniform(seed=0))(X)

            # 创建模型
            model = Model(inputs=X_input, outputs=X, name="ResNet50")

            return model

        model = ResNet50(input_shape=(64, 64, 3), classes=6)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes =  load_dataset()

        # Normalize image vectors
        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.

        # Convert training and test labels to one hot matrices
        Y_train =  convert_to_one_hot(Y_train_orig, 6).T
        Y_test =  convert_to_one_hot(Y_test_orig, 6).T

        print("number of training examples = " + str(X_train.shape[0]))
        print("number of test examples = " + str(X_test.shape[0]))
        print("X_train shape: " + str(X_train.shape))
        print("Y_train shape: " + str(Y_train.shape))
        print("X_test shape: " + str(X_test.shape))
        print("Y_test shape: " + str(Y_test.shape))
        #训练模型
        # model.fit(X_train, Y_train, epochs=1, batch_size=32)
        # preds = model.evaluate(X_test, Y_test)
        # print("误差值 = " + str(preds[0]))
        # print("准确率 = " + str(preds[1]))

        #加载模型
        model = load_model('./datasets/L4W2/datasets/ResNet50.h5')
        preds = model.evaluate(X_test, Y_test)
        print("误差值 = " + str(preds[0]))
        print("准确率 = " + str(preds[1]))

        #之前用的evaluate或eval 前者是模型点keras 后者是变量点tensorflow 二者都是用于测试验证集 这里用predict 这是用于预测的
        #预测只是前向传播 输出结果是y 测试训练集返回的是指标不输出预测结果而为了输出准确率还需要输入真实值和预测值比较
        # from PIL import Image
        # import numpy as np
        # import matplotlib.pyplot as plt  # plt 用于显示图片
        # img_path = 'images/fingers_big/2.jpg'
        # my_image = image.load_img(img_path, target_size=(64, 64))
        # my_image = image.img_to_array(my_image)
        # my_image = np.expand_dims(my_image, axis=0)
        # my_image = preprocess_input(my_image)
        # print("my_image.shape = " + str(my_image.shape))
        # print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
        # print(model.predict(my_image))
        # my_image = scipy.misc.imread(img_path)
        # plt.imshow(my_image)
    def car_demo(self):
        """
        yolov2 实现车辆识别
        Returns:

        """
        import argparse
        import os
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import imshow
        import scipy.io
        import scipy.misc
        import numpy as np
        import pandas as pd
        import PIL
        import tensorflow as tf
        from tensorflow.python.keras import backend as K
        from tensorflow.python.keras.layers import Input, Lambda, Conv2D
        from tensorflow.python.keras.models import load_model, Model

        from datasets.L4W3.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, \
            yolo_body

        from datasets.L4W3 import yolo_utils
        def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):
            """
            通过阈值来过滤对象和分类的置信度。

            参数：
                box_confidence  - tensor类型，维度为（19,19,5,1）,包含19x19单元格中每个单元格预测的5个锚框中的所有的锚框的pc （一些对象的置信概率）。
                boxes - tensor类型，维度为(19,19,5,4)，包含了所有的锚框的（px,py,ph,pw ）。
                box_class_probs - tensor类型，维度为(19,19,5,80)，包含了所有单元格中所有锚框的所有对象( c1,c2,c3，···，c80 )检测的概率。
                threshold - 实数，阈值，如果分类预测的概率高于它，那么这个分类预测的概率就会被保留。

            返回：
                scores - tensor 类型，维度为(None,)，包含了保留了的锚框的分类概率。
                boxes - tensor 类型，维度为(None,4)，包含了保留了的锚框的(b_x, b_y, b_h, b_w)
                classess - tensor 类型，维度为(None,)，包含了保留了的锚框的索引

            注意："None"是因为你不知道所选框的确切数量，因为它取决于阈值。
                  比如：如果有10个锚框，scores的实际输出大小将是（10,）
            """

            # 第一步：计算锚框的得分
            box_scores = box_confidence * box_class_probs

            # 第二步：找到最大值的锚框的索引以及对应的最大值的锚框的分数
            box_classes = K.argmax(box_scores, axis=-1)
            box_class_scores = K.max(box_scores, axis=-1)

            # 第三步：根据阈值创建掩码
            filtering_mask = (box_class_scores >= threshold)

            # 对scores, boxes 以及 classes使用掩码
            scores = tf.boolean_mask(box_class_scores, filtering_mask)
            boxes = tf.boolean_mask(boxes, filtering_mask)
            classes = tf.boolean_mask(box_classes, filtering_mask)

            return scores, boxes, classes

        def iou(box1, box2):
            """
            实现两个锚框的交并比的计算

            参数：
                box1 - 第一个锚框，元组类型，(x1, y1, x2, y2)
                box2 - 第二个锚框，元组类型，(x1, y1, x2, y2)

            返回：
                iou - 实数，交并比。
            """
            # 计算相交的区域的面积
            xi1 = np.maximum(box1[0], box2[0])
            yi1 = np.maximum(box1[1], box2[1])
            xi2 = np.minimum(box1[2], box2[2])
            yi2 = np.minimum(box1[3], box2[3])
            inter_area = (xi1 - xi2) * (yi1 - yi2)

            # 计算并集，公式为：Union(A,B) = A + B - Inter(A,B)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area

            # 计算交并比
            iou = inter_area / union_area

            return iou

        def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
            """
            为锚框实现非最大值抑制（ Non-max suppression (NMS)）

            参数：
                scores - tensor类型，维度为(None,)，yolo_filter_boxes()的输出
                boxes - tensor类型，维度为(None,4)，yolo_filter_boxes()的输出，已缩放到图像大小（见下文）
                classes - tensor类型，维度为(None,)，yolo_filter_boxes()的输出
                max_boxes - 整数，预测的锚框数量的最大值
                iou_threshold - 实数，交并比阈值。

            返回：
                scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
                boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
                classes - tensor类型，维度为(,None)，每个锚框的预测的分类

            注意："None"是明显小于max_boxes的，这个函数也会改变scores、boxes、classes的维度，这会为下一步操作提供方便。

            """
            max_boxes_tensor = K.variable(max_boxes, dtype="int32")  # 用于tf.image.non_max_suppression()
            K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # 初始化变量max_boxes_tensor

            # 使用使用tf.image.non_max_suppression()来获取与我们保留的框相对应的索引列表
            nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

            # 使用K.gather()来选择保留的锚框
            scores = K.gather(scores, nms_indices)
            boxes = K.gather(boxes, nms_indices)
            classes = K.gather(classes, nms_indices)

            return scores, boxes, classes

        def yolo_eval(yolo_outputs, image_shape=(720., 1280.),
                      max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
            """
            将YOLO编码的输出（很多锚框）转换为预测框以及它们的分数，框坐标和类。

            参数：
                yolo_outputs - 编码模型的输出（对于维度为（608,608,3）的图片），包含4个tensors类型的变量：
                                box_confidence ： tensor类型，维度为(None, 19, 19, 5, 1)
                                box_xy         ： tensor类型，维度为(None, 19, 19, 5, 2)
                                box_wh         ： tensor类型，维度为(None, 19, 19, 5, 2)
                                box_class_probs： tensor类型，维度为(None, 19, 19, 5, 80)
                image_shape - tensor类型，维度为（2,），包含了输入的图像的维度，这里是(608.,608.)
                max_boxes - 整数，预测的锚框数量的最大值
                score_threshold - 实数，可能性阈值。
                iou_threshold - 实数，交并比阈值。

            返回：
                scores - tensor类型，维度为(,None)，每个锚框的预测的可能值
                boxes - tensor类型，维度为(4,None)，预测的锚框的坐标
                classes - tensor类型，维度为(,None)，每个锚框的预测的分类
            """

            # 获取YOLO模型的输出
            box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

            # 中心点转换为边角
            boxes = yolo_boxes_to_corners(box_xy, box_wh)

            # 可信度分值过滤
            scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

            # 缩放锚框，以适应原始图像
            boxes = yolo_utils.scale_boxes(boxes, image_shape)

            # 使用非最大值抑制
            scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

            return scores, boxes, classes

        sess = K.get_session()
        class_names = yolo_utils.read_classes("./datasets/L4W3/model_data/coco_classes.txt")
        anchors = yolo_utils.read_anchors("./datasets/L4W3/model_data/yolo_anchors.txt")
        image_shape = (720., 1280.)
        yolo_model = load_model("./datasets/L4W3/model_data/yolov2.h5")
        yolo_model.summary()
        yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
        scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

        def predict(sess, image_file, is_show_info=True, is_plot=True):
            """
            运行存储在sess的计算图以预测image_file的边界框，打印出预测的图与信息。

            参数：
                sess - 包含了YOLO计算图的TensorFlow/Keras的会话。
                image_file - 存储在images文件夹下的图片名称
            返回：
                out_scores - tensor类型，维度为(None,)，锚框的预测的可能值。
                out_boxes - tensor类型，维度为(None,4)，包含了锚框位置信息。
                out_classes - tensor类型，维度为(None,)，锚框的预测的分类索引。
            """
            # 图像预处理
            image, image_data = yolo_utils.preprocess_image("./datasets/L4W3/images/" + image_file, model_image_size=(608, 608))

            # 运行会话并在feed_dict中选择正确的占位符.
            out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                          feed_dict={yolo_model.input: image_data,
                                                                     K.learning_phase(): 0})

            # 打印预测信息
            if is_show_info:
                print("在" + str(image_file) + "中找到了" + str(len(out_boxes)) + "个锚框。")

            # 指定要绘制的边界框的颜色
            colors = yolo_utils.generate_colors(class_names)

            # 在图中绘制边界框
            yolo_utils.draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

            # 保存已经绘制了边界框的图
            image.save(os.path.join("./datasets/L4W3/out", image_file), quality=100)

            # 打印出已经绘制了边界框的图

            import imageio

            if is_plot:
                output_image = imageio.v2.imread(os.path.join("./datasets/L4W3/out", image_file))
                plt.imshow(output_image)

            return out_scores, out_boxes, out_classes

        out_scores, out_boxes, out_classes = predict(sess, "test.jpg")

    def facial_identification_demo(self):
        """
        构建人脸识别系统。
        Returns:

        """


        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
        from tensorflow.python.keras.models import Model
        from tensorflow.python.keras.layers.normalization import BatchNormalization
        from tensorflow.python.keras.layers.pooling import MaxPooling2D, AveragePooling2D
        from tensorflow.python.keras.layers.merge import Concatenate
        from tensorflow.python.keras.layers.core import Lambda, Flatten, Dense
        from tensorflow.python.keras.initializers import glorot_uniform
        from tensorflow.python.keras.layers  import Layer
        from tensorflow.python.keras import backend as K

        K.set_image_data_format('channels_first')

        import cv2
        import os
        import numpy as np
        from numpy import genfromtxt
        import pandas as pd
        import tensorflow as tf
        from datasets.L4W4.facial_identification import fr_utils
        from datasets.L4W4.facial_identification import inception_blocks_v2
        import sys
        np.set_printoptions(threshold=sys.maxsize)

        def triplet_loss(y_true, y_pred, alpha=0.2):
            """
            Implementation of the triplet loss as defined by formula (3)

            Arguments:
            y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
            y_pred -- python list containing three objects:
                    anchor -- the encodings for the anchor images, of shape (None, 128)
                    positive -- the encodings for the positive images, of shape (None, 128)
                    negative -- the encodings for the negative images, of shape (None, 128)

            Returns:
            loss -- real number, value of the loss
            """

            anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

            ### START CODE HERE ### (≈ 4 lines)
            # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
            pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))  # ,axis=-1 528.143
            # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
            neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
            # Step 3: subtract the two previous distances and add alpha.
            basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
            # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
            loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
            ### END CODE HERE ###
            # 如果加上axis=-1，计算出的loss就和expected output相同，但是不能通过作业验证
            return loss

        def verify(image_path, identity, database, model):
            """
            Function that verifies if the person on the "image_path" image is "identity".

            Arguments:
            image_path -- path to an image
            identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
            database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
            model -- your Inception model instance in Keras

            Returns:
            dist -- distance between the image_path and the image of "identity" in the database.
            door_open -- True, if the door should open. False otherwise.
            """

            ### START CODE HERE ###

            # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
            encoding = fr_utils.img_to_encoding(image_path, model)

            # Step 2: Compute distance with identity's image (≈ 1 line)
            dist = np.linalg.norm(encoding - database[identity])

            # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
            if dist < 0.7:
                print("It's " + str(identity) + ", welcome home!")
                door_open = True
            else:
                print("It's not " + str(identity) + ", please go away")
                door_open = False

            ### END CODE HERE ###

            return dist, door_open

        FRmodel = inception_blocks_v2.faceRecoModel(input_shape=(3, 96, 96))
        FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        fr_utils.load_weights_from_FaceNet(FRmodel)

        database = {}
        database["danielle"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/danielle.png", FRmodel)
        database["younes"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/younes.jpg", FRmodel)
        database["tian"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/tian.jpg", FRmodel)
        database["andrew"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/andrew.jpg", FRmodel)
        database["kian"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/kian.jpg", FRmodel)
        database["dan"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/dan.jpg", FRmodel)
        database["sebastiano"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/sebastiano.jpg", FRmodel)
        database["bertrand"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/bertrand.jpg", FRmodel)
        database["kevin"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/kevin.jpg", FRmodel)
        database["felix"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/felix.jpg", FRmodel)
        database["benoit"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/benoit.jpg", FRmodel)
        database["arnaud"] = fr_utils.img_to_encoding("./datasets/L4W4/facial_identification/images/arnaud.jpg", FRmodel)

        verify("./datasets/L4W4/facial_identification/images/camera_0.jpg", "younes", database, FRmodel)

        def who_is_it(image_path, database, model):
            """
            Implements face recognition for the happy house by finding who is the person on the image_path image.

            Arguments:
            image_path -- path to an image
            database -- database containing image encodings along with the name of the person on the image
            model -- your Inception model instance in Keras

            Returns:
            min_dist -- the minimum distance between image_path encoding and the encodings from the database
            identity -- string, the name prediction for the person on image_path
            """

            ### START CODE HERE ###

            ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
            encoding =  fr_utils.img_to_encoding(image_path, model)

            ## Step 2: Find the closest encoding ##

            # Initialize "min_dist" to a large value, say 100 (≈1 line)
            min_dist = 100

            # Loop over the database dictionary's names and encodings.
            for (name, db_enc) in database.items():

                # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
                dist = np.linalg.norm(encoding - db_enc)

                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
                if dist < min_dist:
                    min_dist = dist
                    identity = name

            ### END CODE HERE ###

            if min_dist > 0.7:
                print("Not in the database.")
            else:
                print("it's " + str(identity) + ", the distance is " + str(min_dist))

            return min_dist, identity

        who_is_it("./datasets/L4W4/facial_identification/images/camera_0.jpg", database, FRmodel)

    def style_conversion(self):
        import os
        import sys
        import scipy.io
        import scipy.misc
        import matplotlib.pyplot as plt

        from matplotlib.pyplot import imshow
        from PIL import Image
        import datasets.L4W4.style_conversion.nst_utils
        import numpy as np
        import tensorflow as tf

        def compute_content_cost(a_C, a_G):
            """
            计算内容代价的函数

            参数：
                a_C -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像C的内容的激活值。
                a_G -- tensor类型，维度为(1, n_H, n_W, n_C)，表示隐藏层中图像G的内容的激活值。

            返回：
                J_content -- 实数，用上面的公式1计算的值。

            """

            # 获取a_G的维度信息
            m, n_H, n_W, n_C = a_G.get_shape().as_list()

            # 对a_C与a_G从3维降到2维
            a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
            a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

            # 计算内容代价
            # J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
            J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
            return J_content

        def gram_matrix(A):
            """
            Argument:
            A -- matrix of shape (n_C, n_H*n_W)

            Returns:
            GA -- Gram matrix of A, of shape (n_C, n_C)
            """

            ### START CODE HERE ### (≈1 line)
            GA = tf.matmul(A, tf.transpose(A))
            ### END CODE HERE ###

            return GA

        def compute_layer_style_cost(a_S, a_G):
            """
            Arguments:
            a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
            a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

            Returns:
            J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
            """

            ### START CODE HERE ###
            # Retrieve dimensions from a_G (≈1 line)
            m, n_H, n_W, n_C = a_G.get_shape().as_list()

            # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
            a_S = tf.reshape(a_S, shape=(n_H * n_W, n_C))
            a_G = tf.reshape(a_G, shape=(n_H * n_W, n_C))

            # Computing gram_matrices for both images S and G (≈2 lines)
            GS = gram_matrix(tf.transpose(a_S))
            GG = gram_matrix(tf.transpose(a_G))

            # Computing the loss (≈1 line)
            J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (
                        4 * (n_C * n_C) * (n_W * n_H) * (n_W * n_H))

            ### END CODE HERE ###

            return J_style_layer

        def compute_style_cost(model, STYLE_LAYERS):
            """
            Computes the overall style cost from several chosen layers

            Arguments:
            model -- our tensorflow model
            STYLE_LAYERS -- A python list containing:
                                - the names of the layers we would like to extract style from
                                - a coefficient for each of them

            Returns:
            J_style -- tensor representing a scalar value, style cost defined above by equation (2)
            """
            with tf.Session() as sess:
                # initialize the overall style cost
                J_style = 0

                for layer_name, coeff in STYLE_LAYERS:
                    # Select the output tensor of the currently selected layer
                    out = model[layer_name]

                    # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
                    a_S = sess.run(out)
    
                    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
                    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
                    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
                    a_G = out

                    # Compute style_cost for the current layer
                    J_style_layer = compute_layer_style_cost(a_S, a_G)

                    # Add coeff * J_style_layer of this layer to overall style cost
                    J_style += coeff * J_style_layer

            return J_style

        def total_cost(J_content, J_style, alpha=10, beta=40):
            """
            Computes the total cost function

            Arguments:
            J_content -- content cost coded above
            J_style -- style cost coded above
            alpha -- hyperparameter weighting the importance of the content cost
            beta -- hyperparameter weighting the importance of the style cost

            Returns:
            J -- total cost as defined by the formula above.
            """

            ### START CODE HERE ### (≈1 line)
            J = alpha * J_content + beta * J_style
            ### END CODE HERE ###

            return J
        model = datasets.L4W4.style_conversion.nst_utils.load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
        print(model)
        #model["input"].assign(image)
        content_image = datasets.L4W4.style_conversion.nst_utils.scipy.misc.imread("images/louvre.jpg")
        datasets.L4W4.style_conversion.nst_utils.imshow(content_image)

        style_image = datasets.L4W4.style_conversion.nst_utils.scipy.misc.imread("images/monet_800600.jpg")

        datasets.L4W4.style_conversion.nst_utils.imshow(style_image)

        STYLE_LAYERS = [
            ('conv1_1', 0.2),
            ('conv2_1', 0.2),
            ('conv3_1', 0.2),
            ('conv4_1', 0.2),
            ('conv5_1', 0.2)]
        style_image = datasets.L4W4.style_conversion.nst_utils.scipy.misc.imread("images/monet.jpg")
        style_image = datasets.L4W4.style_conversion.nst_utils.reshape_and_normalize_image(style_image)
        generated_image = datasets.L4W4.style_conversion.nst_utils.generate_noise_image(content_image)
        datasets.L4W4.style_conversion.nst_utils.imshow(generated_image[0])
        model = datasets.L4W4.style_conversion.nst_utils.load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
        # Assign the content image to be the input of the VGG model.
        with tf.Session() as sess:
            sess.run(model['input'].assign(content_image))

        # Select the output tensor of layer conv4_2
        out = model['conv4_2']

        # Set a_C to be the hidden layer activation from the layer we have selected
        a_C = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Assign the input of the model to be the "style" image
        sess.run(model['input'].assign(style_image))

        # Compute the style cost
        J_style = compute_style_cost(model, STYLE_LAYERS)
        ### START CODE HERE ### (1 line)
        J = total_cost(J_content, J_style, alpha=10, beta=40)
        ### END CODE HERE ###
        # define optimizer (1 line)
        optimizer = tf.train.AdamOptimizer(2.0)

        # define train_step (1 line)
        train_step = optimizer.minimize(J)

        def model_nn(sess, input_image, num_iterations=200):

            # Initialize global variables (you need to run the session on the initializer)
            ### START CODE HERE ### (1 line)
            sess.run(tf.global_variables_initializer())
            ### END CODE HERE ###

            # Run the noisy input image (initial generated image) through the model. Use assign().
            ### START CODE HERE ### (1 line)
            generated_image = sess.run(model['input'].assign(input_image))
            ### END CODE HERE ###

            for i in range(num_iterations):

                # Run the session on the train_step to minimize the total cost
                ### START CODE HERE ### (1 line)
                sess.run(train_step)
                ### END CODE HERE ###

                # Compute the generated image by running the session on the current model['input']
                ### START CODE HERE ### (1 line)
                generated_image = sess.run(model['input'])
                ### END CODE HERE ###

                # Print every 20 iteration.
                if i % 20 == 0:
                    Jt, Jc, Js = sess.run([J, J_content, J_style])
                    print("Iteration " + str(i) + " :")
                    print("total cost = " + str(Jt))
                    print("content cost = " + str(Jc))
                    print("style cost = " + str(Js))

                    # save current generated image in the "/output" directory
                    datasets.L4W4.style_conversion.nst_utils.save_image("output/" + str(i) + ".png", generated_image)

            # save last generated image
            datasets.L4W4.style_conversion.nst_utils.save_image('output/generated_image.jpg', generated_image)

            return generated_image

        model_nn(sess, generated_image)

    def rnn_demo(self):
        import numpy as np
        from datasets.L5W1 import rnn_utils

        def rnn_cell_forward(xt, a_prev, parameters):
            """
            根据图2实现RNN单元的单步前向传播

            参数：
                xt -- 时间步“t”输入的数据，维度为（n_x, m）
                a_prev -- 时间步“t - 1”的隐藏隐藏状态，维度为（n_a, m）
                parameters -- 字典，包含了以下内容:
                                Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                                Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                                Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                                ba  -- 偏置，维度为（n_a, 1）
                                by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

            返回：
                a_next -- 下一个隐藏状态，维度为（n_a， m）
                yt_pred -- 在时间步“t”的预测，维度为（n_y， m）
                cache -- 反向传播需要的元组，包含了(a_next, a_prev, xt, parameters)
            """

            # 从“parameters”获取参数
            Wax = parameters["Wax"]
            Waa = parameters["Waa"]
            Wya = parameters["Wya"]
            ba = parameters["ba"]
            by = parameters["by"]

            # 使用上面的公式计算下一个激活值
            a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

            # 使用上面的公式计算当前单元的输出
            yt_pred = rnn_utils.softmax(np.dot(Wya, a_next) + by)

            # 保存反向传播需要的值
            cache = (a_next, a_prev, xt, parameters)

            return a_next, yt_pred, cache

        np.random.seed(1)
        xt = np.random.randn(3, 10)
        a_prev = np.random.randn(5, 10)
        Waa = np.random.randn(5, 5)
        Wax = np.random.randn(5, 3)
        Wya = np.random.randn(2, 5)
        ba = np.random.randn(5, 1)
        by = np.random.randn(2, 1)
        parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

        a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
        print("a_next[4] = ", a_next[4])
        print("a_next.shape = ", a_next.shape)
        print("yt_pred[1] =", yt_pred[1])
        print("yt_pred.shape = ", yt_pred.shape)

        def rnn_forward(x, a0, parameters):
            """
            根据图3来实现循环神经网络的前向传播

            参数：
                x -- 输入的全部数据，维度为(n_x, m, T_x)
                a0 -- 初始化隐藏状态，维度为 (n_a, m)
                parameters -- 字典，包含了以下内容:
                                Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                                Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                                Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                                ba  -- 偏置，维度为（n_a, 1）
                                by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

            返回：
                a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
                y_pred -- 所有时间步的预测，维度为(n_y, m, T_x)
                caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
            """

            # 初始化“caches”，它将以列表类型包含所有的cache
            caches = []

            # 获取 x 与 Wya 的维度信息
            n_x, m, T_x = x.shape
            n_y, n_a = parameters["Wya"].shape

            # 使用0来初始化“a” 与“y”
            a = np.zeros([n_a, m, T_x])
            y_pred = np.zeros([n_y, m, T_x])

            # 初始化“next”
            a_next = a0

            # 遍历所有时间步
            for t in range(T_x):
                ## 1.使用rnn_cell_forward函数来更新“next”隐藏状态与cache。
                a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)

                ## 2.使用 a 来保存“next”隐藏状态（第 t ）个位置。
                a[:, :, t] = a_next

                ## 3.使用 y 来保存预测值。
                y_pred[:, :, t] = yt_pred

                ## 4.把cache保存到“caches”列表中。
                caches.append(cache)

            # 保存反向传播所需要的参数
            caches = (caches, x)

            return a, y_pred, caches

        np.random.seed(1)
        x = np.random.randn(3, 10, 4)
        a0 = np.random.randn(5, 10)
        Waa = np.random.randn(5, 5)
        Wax = np.random.randn(5, 3)
        Wya = np.random.randn(2, 5)
        ba = np.random.randn(5, 1)
        by = np.random.randn(2, 1)
        parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

        a, y_pred, caches = rnn_forward(x, a0, parameters)
        print("a[4][1] = ", a[4][1])
        print("a.shape = ", a.shape)
        print("y_pred[1][3] =", y_pred[1][3])
        print("y_pred.shape = ", y_pred.shape)
        print("caches[1][1][3] =", caches[1][1][3])
        print("len(caches) = ", len(caches))

        def lstm_cell_forward(xt, a_prev, c_prev, parameters):
            """
            根据图4实现一个LSTM单元的前向传播。

            参数：
                xt -- 在时间步“t”输入的数据，维度为(n_x, m)
                a_prev -- 上一个时间步“t-1”的隐藏状态，维度为(n_a, m)
                c_prev -- 上一个时间步“t-1”的记忆状态，维度为(n_a, m)
                parameters -- 字典类型的变量，包含了：
                                Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                                bf -- 遗忘门的偏置，维度为(n_a, 1)
                                Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                                bi -- 更新门的偏置，维度为(n_a, 1)
                                Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                                bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                                Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                                bo -- 输出门的偏置，维度为(n_a, 1)
                                Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                                by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)
            返回：
                a_next -- 下一个隐藏状态，维度为(n_a, m)
                c_next -- 下一个记忆状态，维度为(n_a, m)
                yt_pred -- 在时间步“t”的预测，维度为(n_y, m)
                cache -- 包含了反向传播所需要的参数，包含了(a_next, c_next, a_prev, c_prev, xt, parameters)

            注意：
                ft/it/ot表示遗忘/更新/输出门，cct表示候选值(c tilda)，c表示记忆值。
            """

            # 从“parameters”中获取相关值
            Wf = parameters["Wf"]
            bf = parameters["bf"]
            Wi = parameters["Wi"]
            bi = parameters["bi"]
            Wc = parameters["Wc"]
            bc = parameters["bc"]
            Wo = parameters["Wo"]
            bo = parameters["bo"]
            Wy = parameters["Wy"]
            by = parameters["by"]

            # 获取 xt 与 Wy 的维度信息
            n_x, m = xt.shape
            n_y, n_a = Wy.shape

            # 1.连接 a_prev 与 xt
            contact = np.zeros([n_a + n_x, m])
            contact[: n_a, :] = a_prev
            contact[n_a:, :] = xt

            # 2.根据公式计算ft、it、cct、c_next、ot、a_next

            ## 遗忘门，公式1
            ft = rnn_utils.sigmoid(np.dot(Wf, contact) + bf)

            ## 更新门，公式2
            it = rnn_utils.sigmoid(np.dot(Wi, contact) + bi)

            ## 更新单元，公式3
            cct = np.tanh(np.dot(Wc, contact) + bc)

            ## 更新单元，公式4
            # c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
            c_next = ft * c_prev + it * cct
            ## 输出门，公式5
            ot = rnn_utils.sigmoid(np.dot(Wo, contact) + bo)

            ## 输出门，公式6
            # a_next = np.multiply(ot, np.tan(c_next))
            a_next = ot * np.tanh(c_next)
            # 3.计算LSTM单元的预测值
            yt_pred = rnn_utils.softmax(np.dot(Wy, a_next) + by)

            # 保存包含了反向传播所需要的参数
            cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

            return a_next, c_next, yt_pred, cache

        np.random.seed(1)
        xt = np.random.randn(3, 10)
        a_prev = np.random.randn(5, 10)
        c_prev = np.random.randn(5, 10)
        Wf = np.random.randn(5, 5 + 3)
        bf = np.random.randn(5, 1)
        Wi = np.random.randn(5, 5 + 3)
        bi = np.random.randn(5, 1)
        Wo = np.random.randn(5, 5 + 3)
        bo = np.random.randn(5, 1)
        Wc = np.random.randn(5, 5 + 3)
        bc = np.random.randn(5, 1)
        Wy = np.random.randn(2, 5)
        by = np.random.randn(2, 1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc,
                      "by": by}

        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)
        print("a_next[4] = ", a_next[4])
        print("a_next.shape = ", c_next.shape)
        print("c_next[2] = ", c_next[2])
        print("c_next.shape = ", c_next.shape)
        print("yt[1] =", yt[1])
        print("yt.shape = ", yt.shape)
        print("cache[1][3] =", cache[1][3])
        print("len(cache) = ", len(cache))

        def lstm_forward(x, a0, parameters):
            """
            根据图5来实现LSTM单元组成的的循环神经网络

            参数：
                x -- 所有时间步的输入数据，维度为(n_x, m, T_x)
                a0 -- 初始化隐藏状态，维度为(n_a, m)
                parameters -- python字典，包含了以下参数：
                                Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                                bf -- 遗忘门的偏置，维度为(n_a, 1)
                                Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                                bi -- 更新门的偏置，维度为(n_a, 1)
                                Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                                bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                                Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                                bo -- 输出门的偏置，维度为(n_a, 1)
                                Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                                by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)

            返回：
                a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
                y -- 所有时间步的预测值，维度为(n_y, m, T_x)
                caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
            """

            # 初始化“caches”
            caches = []

            # 获取 xt 与 Wy 的维度信息
            n_x, m, T_x = x.shape
            n_y, n_a = parameters["Wy"].shape

            # 使用0来初始化“a”、“c”、“y”
            a = np.zeros([n_a, m, T_x])
            c = np.zeros([n_a, m, T_x])
            y = np.zeros([n_y, m, T_x])

            # 初始化“a_next”、“c_next”
            a_next = a0
            c_next = np.zeros([n_a, m])

            # 遍历所有的时间步
            for t in range(T_x):
                # 更新下一个隐藏状态，下一个记忆状态，计算预测值，获取cache
                a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)

                # 保存新的下一个隐藏状态到变量a中
                a[:, :, t] = a_next

                # 保存预测值到变量y中
                y[:, :, t] = yt_pred

                # 保存下一个单元状态到变量c中
                c[:, :, t] = c_next

                # 把cache添加到caches中
                caches.append(cache)

            # 保存反向传播需要的参数
            caches = (caches, x)

            return a, y, c, caches

        np.random.seed(1)
        x = np.random.randn(3, 10, 7)
        a0 = np.random.randn(5, 10)
        Wf = np.random.randn(5, 5 + 3)
        bf = np.random.randn(5, 1)
        Wi = np.random.randn(5, 5 + 3)
        bi = np.random.randn(5, 1)
        Wo = np.random.randn(5, 5 + 3)
        bo = np.random.randn(5, 1)
        Wc = np.random.randn(5, 5 + 3)
        bc = np.random.randn(5, 1)
        Wy = np.random.randn(2, 5)
        by = np.random.randn(2, 1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc,
                      "by": by}

        a, y, c, caches = lstm_forward(x, a0, parameters)
        print("a[4][3][6] = ", a[4][3][6])
        print("a.shape = ", a.shape)
        print("y[1][4][3] =", y[1][4][3])
        print("y.shape = ", y.shape)
        print("caches[1][1[1]] =", caches[1][1][1])
        print("c[1][2][1]", c[1][2][1])
        print("len(caches) = ", len(caches))

        def rnn_cell_backward(da_next, cache):
            """
            实现基本的RNN单元的单步反向传播

            参数：
                da_next -- 关于下一个隐藏状态的损失的梯度。
                cache -- 字典类型，rnn_step_forward()的输出

            返回：
                gradients -- 字典，包含了以下参数：
                                dx -- 输入数据的梯度，维度为(n_x, m)
                                da_prev -- 上一隐藏层的隐藏状态，维度为(n_a, m)
                                dWax -- 输入到隐藏状态的权重的梯度，维度为(n_a, n_x)
                                dWaa -- 隐藏状态到隐藏状态的权重的梯度，维度为(n_a, n_a)
                                dba -- 偏置向量的梯度，维度为(n_a, 1)
            """
            # 获取cache 的值
            a_next, a_prev, xt, parameters = cache

            # 从 parameters 中获取参数
            Wax = parameters["Wax"]
            Waa = parameters["Waa"]
            Wya = parameters["Wya"]
            ba = parameters["ba"]
            by = parameters["by"]

            # 计算tanh相对于a_next的梯度.
            dtanh = (1 - np.square(a_next)) * da_next

            # 计算关于Wax损失的梯度
            dxt = np.dot(Wax.T, dtanh)
            dWax = np.dot(dtanh, xt.T)

            # 计算关于Waa损失的梯度
            da_prev = np.dot(Waa.T, dtanh)
            dWaa = np.dot(dtanh, a_prev.T)

            # 计算关于b损失的梯度
            dba = np.sum(dtanh, keepdims=True, axis=-1)

            # 保存这些梯度到字典内
            gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

            return gradients

        np.random.seed(1)
        xt = np.random.randn(3, 10)
        a_prev = np.random.randn(5, 10)
        Wax = np.random.randn(5, 3)
        Waa = np.random.randn(5, 5)
        Wya = np.random.randn(2, 5)
        b = np.random.randn(5, 1)
        by = np.random.randn(2, 1)
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

        a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)

        da_next = np.random.randn(5, 10)
        gradients = rnn_cell_backward(da_next, cache)
        print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
        print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
        print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
        print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
        print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
        print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
        print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
        print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
        print("gradients[\"dba\"][4] =", gradients["dba"][4])
        print("gradients[\"dba\"].shape =", gradients["dba"].shape)

        def rnn_backward(da, caches):
            """
            在整个输入数据序列上实现RNN的反向传播

            参数：
                da -- 所有隐藏状态的梯度，维度为(n_a, m, T_x)
                caches -- 包含向前传播的信息的元组

            返回：
                gradients -- 包含了梯度的字典：
                                dx -- 关于输入数据的梯度，维度为(n_x, m, T_x)
                                da0 -- 关于初始化隐藏状态的梯度，维度为(n_a, m)
                                dWax -- 关于输入权重的梯度，维度为(n_a, n_x)
                                dWaa -- 关于隐藏状态的权值的梯度，维度为(n_a, n_a)
                                dba -- 关于偏置的梯度，维度为(n_a, 1)
            """
            # 从caches中获取第一个cache（t=1）的值
            caches, x = caches
            a1, a0, x1, parameters = caches[0]

            # 获取da与x1的维度信息
            n_a, m, T_x = da.shape
            n_x, m = x1.shape

            # 初始化梯度
            dx = np.zeros([n_x, m, T_x])
            dWax = np.zeros([n_a, n_x])
            dWaa = np.zeros([n_a, n_a])
            dba = np.zeros([n_a, 1])
            da0 = np.zeros([n_a, m])
            da_prevt = np.zeros([n_a, m])

            # 处理所有时间步
            for t in reversed(range(T_x)):
                # 计算时间步“t”时的梯度
                gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])

                # 从梯度中获取导数
                dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], \
                gradients["dWaa"], gradients["dba"]

                # 通过在时间步t添加它们的导数来增加关于全局导数的参数
                dx[:, :, t] = dxt
                dWax += dWaxt
                dWaa += dWaat
                dba += dbat

            # 将 da0设置为a的梯度，该梯度已通过所有时间步骤进行反向传播
            da0 = da_prevt

            # 保存这些梯度到字典内
            gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

            return gradients

        np.random.seed(1)
        x = np.random.randn(3, 10, 4)
        a0 = np.random.randn(5, 10)
        Wax = np.random.randn(5, 3)
        Waa = np.random.randn(5, 5)
        Wya = np.random.randn(2, 5)
        ba = np.random.randn(5, 1)
        by = np.random.randn(2, 1)
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
        a, y, caches = rnn_forward(x, a0, parameters)
        da = np.random.randn(5, 10, 4)
        gradients = rnn_backward(da, caches)

        print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
        print("gradients[\"dx\"].shape =", gradients["dx"].shape)
        print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
        print("gradients[\"da0\"].shape =", gradients["da0"].shape)
        print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
        print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
        print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
        print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
        print("gradients[\"dba\"][4] =", gradients["dba"][4])
        print("gradients[\"dba\"].shape =", gradients["dba"].shape)

        def lstm_cell_backward(da_next, dc_next, cache):
            """
            实现LSTM的单步反向传播

            参数：
                da_next -- 下一个隐藏状态的梯度，维度为(n_a, m)
                dc_next -- 下一个单元状态的梯度，维度为(n_a, m)
                cache -- 来自前向传播的一些参数

            返回：
                gradients -- 包含了梯度信息的字典：
                                dxt -- 输入数据的梯度，维度为(n_x, m)
                                da_prev -- 先前的隐藏状态的梯度，维度为(n_a, m)
                                dc_prev -- 前的记忆状态的梯度，维度为(n_a, m, T_x)
                                dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                                dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                                dWi -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                                dbi -- 更新门的偏置的梯度，维度为(n_a, 1)
                                dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                                dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                                dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                                dbo -- 输出门的偏置的梯度，维度为(n_a, 1)
            """
            # 从cache中获取信息
            (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

            # 获取xt与a_next的维度信息
            n_x, m = xt.shape
            n_a, m = a_next.shape

            # 根据公式7-10来计算门的导数
            dot = da_next * np.tanh(c_next) * ot * (1 - ot)
            dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
            dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
            dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

            # 根据公式11-14计算参数的导数
            concat = np.concatenate((a_prev, xt), axis=0).T
            dWf = np.dot(dft, concat)
            dWi = np.dot(dit, concat)
            dWc = np.dot(dcct, concat)
            dWo = np.dot(dot, concat)
            dbf = np.sum(dft, axis=1, keepdims=True)
            dbi = np.sum(dit, axis=1, keepdims=True)
            dbc = np.sum(dcct, axis=1, keepdims=True)
            dbo = np.sum(dot, axis=1, keepdims=True)

            # 使用公式15-17计算洗起来了隐藏状态、先前记忆状态、输入的导数。
            da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) + np.dot(
                parameters["Wi"][:, :n_a].T, dit) + np.dot(parameters["Wo"][:, :n_a].T, dot)

            dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next

            dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) + np.dot(
                parameters["Wi"][:, n_a:].T, dit) + np.dot(parameters["Wo"][:, n_a:].T, dot)

            # 保存梯度信息到字典
            gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi,
                         "dbi": dbi,
                         "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

            return gradients

        np.random.seed(1)
        xt = np.random.randn(3, 10)
        a_prev = np.random.randn(5, 10)
        c_prev = np.random.randn(5, 10)
        Wf = np.random.randn(5, 5 + 3)
        bf = np.random.randn(5, 1)
        Wi = np.random.randn(5, 5 + 3)
        bi = np.random.randn(5, 1)
        Wo = np.random.randn(5, 5 + 3)
        bo = np.random.randn(5, 1)
        Wc = np.random.randn(5, 5 + 3)
        bc = np.random.randn(5, 1)
        Wy = np.random.randn(2, 5)
        by = np.random.randn(2, 1)

        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc,
                      "by": by}

        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

        da_next = np.random.randn(5, 10)
        dc_next = np.random.randn(5, 10)
        gradients = lstm_cell_backward(da_next, dc_next, cache)
        print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
        print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
        print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
        print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
        print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
        print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
        print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
        print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
        print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
        print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
        print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
        print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
        print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
        print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
        print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
        print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
        print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
        print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
        print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
        print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
        print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
        print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)
    def dinos_demo(self):
        """
        通过循环神经网络实现，字符级语言模型 - 恐龙岛
        欢迎来到恐龙岛，恐龙生活于在6500万年前，现在研究人员在试着复活恐龙，而你的任务就是给恐龙命名，
        如果一只恐龙不喜欢它的名字，它可能会狂躁不安，所以你要谨慎选择。
        你的助手已经收集了他们能够找到的所有恐龙名字，并编入了这个数据集,
        为了构建字符级语言模型来生成新的名称，你的模型将学习不同的名称模式，
        并随机生成新的名字。希望这个算法能让你和你的团队远离恐龙的愤怒。

        在这里你将学习到：
        如何存储文本数据以便使用RNN进行处理。
        如何合成数据，通过每次采样预测，并将其传递给下一个rnn单元。
        如何构建字符级文本生成循环神经网络。
        为什么梯度修剪很重要?
        Returns:

        """
        import numpy as np
        import random
        import time
        from datasets.L5W1 import cllm_utils
        # 获取名称
        data = open("datasets/L5W1/dinos.txt", "r").read()
        # 转化为小写字符
        data = data.lower()
        # 转化为无序且不重复的元素列表
        chars = list(set(data))

        # 获取大小信息
        data_size, vocab_size = len(data), len(chars)
        print(chars)
        print("共计有%d个字符，唯一字符有%d个" % (data_size, vocab_size))
        char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
        ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

        print(char_to_ix)
        print(ix_to_char)

        def clip(gradients, maxValue):
            """
            使用maxValue来修剪梯度

            参数：
                gradients -- 字典类型，包含了以下参数："dWaa", "dWax", "dWya", "db", "dby"
                maxValue -- 阈值，把梯度值限制在[-maxValue, maxValue]内

            返回：
                gradients -- 修剪后的梯度
            """
            # 获取参数
            dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], \
            gradients['dby']

            # 梯度修剪
            for gradient in [dWaa, dWax, dWya, db, dby]:
                np.clip(gradient, -maxValue, maxValue, out=gradient)

            gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

            return gradients

        def sample(parameters, char_to_is, seed):
            """
            根据RNN输出的概率分布序列对字符序列进行采样

            参数：
                parameters -- 包含了Waa, Wax, Wya, by, b的字典
                char_to_ix -- 字符映射到索引的字典
                seed -- 随机种子

            返回：
                indices -- 包含采样字符索引的长度为n的列表。
            """

            # 从parameters 中获取参数
            Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], \
            parameters['b']
            vocab_size = by.shape[0]
            n_a = Waa.shape[1]

            # 步骤1
            ## 创建独热向量x
            x = np.zeros((vocab_size, 1))

            ## 使用0初始化a_prev
            a_prev = np.zeros((n_a, 1))

            # 创建索引的空列表，这是包含要生成的字符的索引的列表。
            indices = []

            # IDX是检测换行符的标志，我们将其初始化为-1。
            idx = -1

            # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，
            # 并将其索引附加到“indices”上，如果我们达到50个字符，
            # （我们应该不太可能有一个训练好的模型），我们将停止循环，这有助于调试并防止进入无限循环
            counter = 0
            newline_character = char_to_ix["\n"]

            while (idx != newline_character and counter < 50):
                # 步骤2：使用公式1、2、3进行前向传播
                a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
                z = np.dot(Wya, a) + by
                y = cllm_utils.softmax(z)

                # 设定随机种子
                np.random.seed(counter + seed)

                # 步骤3：从概率分布y中抽取词汇表中字符的索引
                idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

                # 添加到索引中
                indices.append(idx)

                # 步骤4:将输入字符重写为与采样索引对应的字符。
                x = np.zeros((vocab_size, 1))
                x[idx] = 1

                # 更新a_prev为a
                a_prev = a

                # 累加器
                seed += 1
                counter += 1

            if (counter == 50):
                indices.append(char_to_ix["\n"])

            return indices

        np.random.seed(2)
        _, n_a = 20, 100
        Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
        b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

        indices = sample(parameters, char_to_ix, 0)
        print("Sampling:")
        print("list of sampled indices:", indices)
        print("list of sampled characters:", [ix_to_char[i] for i in indices])

        def optimize(X, Y, a_prev, parameters, learning_rate=0.01):
            """
            执行训练模型的单步优化。

            参数：
                X -- 整数列表，其中每个整数映射到词汇表中的字符。
                Y -- 整数列表，与X完全相同，但向左移动了一个索引。
                a_prev -- 上一个隐藏状态
                parameters -- 字典，包含了以下参数：
                                Wax -- 权重矩阵乘以输入，维度为(n_a, n_x)
                                Waa -- 权重矩阵乘以隐藏状态，维度为(n_a, n_a)
                                Wya -- 隐藏状态与输出相关的权重矩阵，维度为(n_y, n_a)
                                b -- 偏置，维度为(n_a, 1)
                                by -- 隐藏状态与输出相关的权重偏置，维度为(n_y, 1)
                learning_rate -- 模型学习的速率

            返回：
                loss -- 损失函数的值（交叉熵损失）
                gradients -- 字典，包含了以下参数：
                                dWax -- 输入到隐藏的权值的梯度，维度为(n_a, n_x)
                                dWaa -- 隐藏到隐藏的权值的梯度，维度为(n_a, n_a)
                                dWya -- 隐藏到输出的权值的梯度，维度为(n_y, n_a)
                                db -- 偏置的梯度，维度为(n_a, 1)
                                dby -- 输出偏置向量的梯度，维度为(n_y, 1)
                a[len(X)-1] -- 最后的隐藏状态，维度为(n_a, 1)
            """

            # 前向传播
            loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters)

            # 反向传播
            gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache)

            # 梯度修剪，[-5 , 5]
            gradients = clip(gradients, 5)

            # 更新参数
            parameters = cllm_utils.update_parameters(parameters, gradients, learning_rate)

            return loss, gradients, a[len(X) - 1]

        np.random.seed(1)
        vocab_size, n_a = 27, 100
        a_prev = np.random.randn(n_a, 1)
        Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
        b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
        parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
        X = [12, 3, 5, 11, 22, 3]
        Y = [4, 14, 11, 22, 25, 26]

        loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate=0.01)
        print("Loss =", loss)
        print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
        print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
        print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
        print("gradients[\"db\"][4] =", gradients["db"][4])
        print("gradients[\"dby\"][1] =", gradients["dby"][1])
        print("a_last[4] =", a_last[4])

        def model(data, ix_to_char, char_to_ix, num_iterations=3500,
                  n_a=50, dino_names=7, vocab_size=27):
            """
            训练模型并生成恐龙名字

            参数：
                data -- 语料库
                ix_to_char -- 索引映射字符字典
                char_to_ix -- 字符映射索引字典
                num_iterations -- 迭代次数
                n_a -- RNN单元数量
                dino_names -- 每次迭代中采样的数量
                vocab_size -- 在文本中的唯一字符的数量

            返回：
                parameters -- 学习后了的参数
            """

            # 从vocab_size中获取n_x、n_y
            n_x, n_y = vocab_size, vocab_size

            # 初始化参数
            parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)

            # 初始化损失
            loss = cllm_utils.get_initial_loss(vocab_size, dino_names)

            # 构建恐龙名称列表
            with open("datasets/L5W1/dinos.txt") as f:
                examples = f.readlines()
            examples = [x.lower().strip() for x in examples]

            # 打乱全部的恐龙名称
            np.random.seed(0)
            np.random.shuffle(examples)

            # 初始化LSTM隐藏状态
            a_prev = np.zeros((n_a, 1))

            # 循环
            for j in range(num_iterations):
                # 定义一个训练样本
                index = j % len(examples)
                X = [None] + [char_to_ix[ch] for ch in examples[index]]
                Y = X[1:] + [char_to_ix["\n"]]

                # 执行单步优化：前向传播 -> 反向传播 -> 梯度修剪 -> 更新参数
                # 选择学习率为0.01
                curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

                # 使用延迟来保持损失平滑,这是为了加速训练。
                loss = cllm_utils.smooth(loss, curr_loss)

                # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
                if j % 2000 == 0:
                    print("第" + str(j + 1) + "次迭代，损失值为：" + str(loss))

                    seed = 0
                    for name in range(dino_names):
                        # 采样
                        sampled_indices = sample(parameters, char_to_ix, seed)
                        cllm_utils.print_sample(sampled_indices, ix_to_char)

                        # 为了得到相同的效果，随机种子+1
                        seed += 1

                    print("\n")
            return parameters

        # 开始时间
        start_time =time.perf_counter()

        # 开始训练
        parameters = model(data, ix_to_char, char_to_ix, num_iterations=3500)

        # 结束时间
        end_time = time.perf_counter()

        # 计算时差
        minium = end_time - start_time

        print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium % 60)) + "秒")


if __name__ == '__main__':
    myexample = Example()
    #myexample.mnist_demo()
    #myexample.cat_demo()
    #myexample.flower_demo()
    #myexample.cat_new_demo()
    #myexample.cat_final_demo()
    #myexample.init_demo()
    #myexample.reg_demo()
    #myexample.gc_demo()
    #myexample.opt_demo()
    #myexample.tensorflow_demo()
    #从此案例开始使用py37 tensorflow1.15.0环境 与gpu不匹配所以会警告dll缺失
    #本机安装的cuda cudnn 是8.0 11.0 对应python3.7 tensorflow==2.4.0 GPU
    #myexample.sign_language_demo()
    #myexample.cnn_demo()
    #myexample.smiling_face()
    #myexample.resnets_demo()
    #myexample.car_demo()
    #myexample.facial_identification_demo()
    #myexample.style_conversion()
    #myexample.rnn_demo()
    myexample.dinos_demo()