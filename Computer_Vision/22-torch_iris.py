from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt

# 读取鸢尾花数据集，并进行切分
x, y = load_iris(return_X_y=True)
# 数据集切分
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

# 设置变量
x_train = torch.autograd.Variable(torch.Tensor(train_x))
x_test = torch.autograd.Variable(torch.Tensor(test_x))
y_train = torch.autograd.Variable(torch.Tensor(train_y))
y_test = torch.autograd.Variable(torch.Tensor(test_y))

# 搭建模型
model = torch.nn.Sequential(
    torch.nn.Linear(4, 3)
)
# 设置代价函数
cost = torch.nn.CrossEntropyLoss()
# 设置优化器（梯度下降）
opti = torch.optim.Adam(model.parameters(), lr=0.1)
# 模型训练
train_loss = []
val_loss = []
for i in range(101):
    # 训练损失计算
    h = model(x_train)
    loss = cost(h, y_train.long())
    train_loss.append(loss)
    loss.backward()
    # 验证损失计算
    h1 = model(x_test)
    loss_ = cost(h1, y_test.long())
    val_loss.append(loss_)
    # -----------------
    opti.step()
    opti.zero_grad()
    if i % 20 == 0:
        pre = torch.max(h, 1)[1]
        acc = torch.mean((y_train==pre).float())
        print(i, acc.data.numpy(), loss.data.numpy())

# 绘制模型评测效果
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(train_loss, color='red', label='训练损失')
plt.plot(val_loss, color='blue', label='测试损失')
plt.legend()
plt.show()