from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy# 二分类交叉熵
import numpy as np

x_data = np.array(
         [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]])
# 二分类模型标签使用0,1表达正负类别
y_data = np.array(
         [[0],
          [0],
          [0],
          [1],
          [1],
          [1]])

model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

sgd = SGD(lr=0.1) # 学习率0.1
# 二分类交叉熵损失
model.compile(loss=binary_crossentropy, optimizer=sgd)

model.summary()
# verbose=0 不显示每次运算结果
model.fit(x_data, y_data, epochs=2000, verbose=0)

# 打印预测结果
print('预测结果：', model.predict(x_data))