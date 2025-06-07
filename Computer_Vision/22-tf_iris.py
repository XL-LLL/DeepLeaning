from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras import utils
import matplotlib.pyplot as plt

# 读取鸢尾花数据集，并进行切分
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)
# 将y标签进行独热编码
y_train = utils.to_categorical(y_train, 3)
y_test = utils.to_categorical(y_test, 3)

# 模型构建
model = models.Sequential()
model.add(layers.Dense(3, input_dim=(4), activation='softmax'))
# 模型配置
model.compile(optimizer=optimizers.Adam(0.01),
              loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])
# 模型训练
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

# 绘制模型评测效果
plt.rcParams['font.sans-serif'] = ['SimHei']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, color='red', label='训练损失')
plt.plot(val_loss, color='blue', label='测试损失')
plt.legend()
plt.show()

