import torch
from torch.autograd import Variable
import numpy as np

# 4个特征，3个类别
x_data = np.array([[1, 2, 1, 1],
                   [2, 1, 3, 2],
                   [3, 1, 3, 4],
                   [4, 1, 5, 5],
                   [1, 7, 5, 5],
                   [1, 2, 5, 6],
                   [1, 6, 6, 6],
                   [1, 7, 7, 7]],
                  dtype=np.float32)
y_data = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0]],
                  dtype=np.float32)

X = Variable(torch.Tensor(x_data))
Y = Variable(torch.Tensor(y_data))

# 创建模型，输入神经元4个，输出神经元3个
linear = torch.nn.Linear(4, 3, bias=True)
model = torch.nn.Sequential(linear)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 与二分类有区别，损失函数和评估指标需要使用多分类类型
# 该函数会调用softmax函数
criterion = torch.nn.CrossEntropyLoss()
for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    # 多分类交叉熵函数
    cost = criterion(hypothesis, torch.argmax(Y, 1).long())
    cost.backward()
    optimizer.step()

    if step % 500 == 0:
        print(step, cost.data.numpy())

# 准确率
h = model(X) # 预测概率
pre = torch.max(h, 1)[1].data.numpy() # 预测类别
y_ = torch.max(Y, 1)[1].data.numpy() # 真实类别
print('最终准确率为：\n', np.mean(pre==y_))


