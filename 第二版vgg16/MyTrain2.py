import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, keras:
    print(module.__name__, module.__version__)

train_dir = '/content/train_local/input/training/training'
validation_dir = '/content/train_local/input/validation/validation'

# 定义一些静态变量，比如图片的尺寸，分批训练的数量等
height = 224
width = 224
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
#方法1
# model = keras.models.Sequential([
#     keras.layers.Conv2D(64, 3,
#         activation='relu',
#        input_shape=[224, 224, 3],
#         padding='same'),
#    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
#    keras.layers.MaxPooling2D(pool_size=2, strides=2),
#    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
#    keras.layers.MaxPooling2D(pool_size=2, strides=2),
#    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
#    keras.layers.MaxPooling2D(pool_size=2, strides=2),
#    keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
#    keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
#    keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
#    keras.layers.MaxPooling2D(pool_size=2, strides=2),
#    keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
#    keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
#    keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
#    keras.layers.MaxPooling2D(pool_size=2, strides=2),
#    keras.layers.Flatten(),
#    keras.layers.Dense(4096, activation='relu'),
#    keras.layers.Dropout(0.5),
#    keras.layers.Dense(4096, activation='relu'),
#    keras.layers.Dropout(0.5),
#    keras.layers.Dense(10, activation='softmax'),
# ], name='VGG-16')
#方法2
#直接加载TF的VGG模型，并采用在ImageNet预训练的权重
model = tf.keras.applications.vgg16.VGG16(
                              weights='imagenet')
#自定义自己的全连接层
covn_base = tf.keras.applications.vgg16.VGG16(
                   weights='imagenet', include_top=False,#不要最后分类层
                   input_shape=(224, 224, 3))
covn_base.trainable = True
for layers in covn_base.layers[:-4]:  # 让前面卷积层不可训练
    layers.trainable = False          # 只让训练后面的4层
model = keras.models.Sequential([
        covn_base,
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax'),
], name='VGG-16')
model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(0.0001), metrics=['accuracy'])
model.summary()
log_dir="/content/drive/MyDrive/tf/input/logs2"
if not os.path.exists(log_dir):
  os.mkdir(log_dir)# 创建保存目录
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,
                    profile_batch = 100000000)
# 训练
#tf.optimizers.Adam(0.0001)，可以换成自己想要的学习率（学习步长）
epochs = 100
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
model.save('/content/drive/MyDrive/tf/input/model2.h5')
print('完毕')
