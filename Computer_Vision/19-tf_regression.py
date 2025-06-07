from tensorflow.keras.models import Sequential # 序列化模式创建模型
from tensorflow.keras.layers import Dense # 创建模型层
from tensorflow.keras.optimizers import Adam # 优化器（梯度下降）
import matplotlib.pyplot as plt

x_data = [[1], [2], [3]]
y_data = [[1], [2], [3]]

# 创建模型
model = Sequential()
# 输入特征数量为1个， 输出神经元1个
model.add(Dense(1, input_dim=(1)))

# 显示模型结构
model.summary()

# 模型配置
model.compile(optimizer=Adam(0.01), loss='mse')

# 模型训练
history = model.fit(x_data, y_data, epochs=400, verbose=0)

# 读取训练过程中的损失值
cost_list = history.history['loss']
# 损失值迭代过程可视化
plt.plot(cost_list)
plt.show()

# 预测
print(model.predict(x_data))

