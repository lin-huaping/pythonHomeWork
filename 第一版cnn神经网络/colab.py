import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, keras:
    print(module.__name__, module.__version__)

train_dir = '/content/train_local/input/training/training'
validation_dir = '/content/train_local/input/validation/validation'

# 定义一些静态变量，比如图片的尺寸，分批训练的数量等
height = 300
width = 400
channels = 3
batch_size = 64
num_classes = 10

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(height, width),
    batch_size=batch_size, seed=7,
    shuffle=True, class_mode='categorical'
)
valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
valid_generator = valid_datagen.flow_from_directory(
    validation_dir, target_size=(height, width),
    batch_size=batch_size, seed=7,
    shuffle=False, class_mode='categorical'
)

train_num = train_generator.samples
valid_num = valid_generator.samples
print(train_num, valid_num)

for i in range(2):
    x, y = train_generator.next()
    print(x.shape, y.shape)
    #print(y)

# 构建模型
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                        activation='relu',
                        input_shape=[width, height, channels]),
    keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                        activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(0.0001), metrics=['accuracy'])
model.summary()
log_dir="/content/drive/MyDrive/tf/input/logs"
if not os.path.exists(log_dir):
  os.mkdir(log_dir)# 创建保存目录
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                    profile_batch = 100000000)
# 训练
epochs = 200
history = model.fit_generator(
    train_generator, steps_per_epoch= train_num // batch_size,
    epochs=epochs, validation_data=valid_generator,
    validation_steps= valid_num // batch_size,
    callbacks=[tensorboard_callback]
)
def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8,5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()
#print(history.history.keys())
plot_learning_curves(history, 'accuracy', epochs, 0, 1)
plot_learning_curves(history, 'loss', epochs, 0, 5)
model.save('/content/drive/MyDrive/tf/input/model.h5')
print('完毕')

#
#这个目前就这样定下来300*400的尺寸，训练200次的结果，学习率调整为0.0001。用的神经网络。200次预计出3个小时来跑，甚至不止。
#在colab上跑，把数据，比如图片进行保存，代码到时就在这看，还有一个网上的图可以看，就这样吧，加油。
#另一个模型有待整改，首先是试试预训练模式，改动原本vgg16的模式。其他不变。ui考虑搞不搞。可以不搞。
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir train_local
# %cp -av /content/drive/MyDrive/tf/input   /content/train_local
# !pip install pyunpack
# !pip install patool
# from pyunpack import Archive
# Archive('/content/train_local/JPEGImages.rar').extractall('/content/train_local')


# LOG_DIR = '/content/drive/MyDrive/tf/input/logs'
# get_ipython().system_raw('tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'.format(LOG_DIR))
# #开启ngrok service，绑定port 6006(tensorboard)
# get_ipython().system_raw('./ngrok http 6006 &')
# ! curl -s http://localhost:4040/api/tunnels | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
# !tensorboard --logdir=/content/drive/MyDrive/tf/input/logs/train
