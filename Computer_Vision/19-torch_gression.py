import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# 判断运行版本的类型
device = torch.device("cpu")

x_data = [[1], [2], [3]]
y_data = [[1], [2], [3]]

# 使用变量存储特征和标签  tensor不能反向传播，variable可以反向传播。
X = Variable(torch.Tensor(x_data))
Y = Variable(torch.Tensor(y_data))

# 数据添加到算力中(GPU或CPU)
X = X.to(device)
Y = Y.to(device)
# 创建模型
model = nn.Linear(1, 1, bias=True)

# 模型加载到GPU中
model.to(device)

# 调用代价函数
criterion = nn.MSELoss()

# 优化器使用 model.parameters(),返回各层参数
# torch.optim.Adam 第一个参数 待优化参数的iterable或者是定义了参数组的dict
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 模型训练处理
epoch = 401
loss_history = np.zeros(epoch)
for step in range(epoch):
    optimizer.zero_grad() # 梯度归零否则每次会叠加
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward() # 反向传播 #梯度累加的处理
    optimizer.step() # 更新参数
    loss_history[step] = cost.data.cpu().numpy()
    if step % 100 == 0:
        print('第{}次循环'.format(step), '代价数值为{}'.format(cost.data.cpu().numpy()))

import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.show()
# 预测结果
print(model(X).data.numpy())

