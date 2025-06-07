from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers, losses, metrics
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

# 创建模型，输入神经元4个，输出神经元3个
# 数据集为多分类类型，使用softmax函数处理多分类
model = Sequential()
model.add(Dense(3, input_shape=(4,)))
model.add(Activation('softmax'))

model.summary()

# 与二分类有区别，损失函数和评估指标需要使用多分类类型
model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adam(0.1),
              metrics=metrics.categorical_accuracy)

model.fit(x_data, y_data, epochs=2000)
