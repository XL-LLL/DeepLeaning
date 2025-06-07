import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F # 调用损失函数

torch.manual_seed(777)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 二分类模型标签使用0,1表达正负类别
x_data = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]], dtype=np.float32)
y_data = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)

X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

# 创建模型后，使用sigmoid函数进行二分类处理
linear = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear, sigmoid)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(2001):
    optimizer.zero_grad()
    hypothesis = model(X)
    # 调用二分类损失函数
    cost = F.binary_cross_entropy(hypothesis, Y)
    cost.backward() # 累加梯度值
    optimizer.step()

    if step % 500 == 0:
        print(step, cost.data.numpy())

# 准确结果预测
predicted = (model(X).data > 0.5).float()
print('预测结果为：', predicted)
