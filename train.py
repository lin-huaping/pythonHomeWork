import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix
train_dir = './input/training/training'
validation_dir = './input/validation/validation'
height = 128
width = 128
valid_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
valid_test= valid_datagen.flow_from_directory(
    validation_dir, target_size=(height, width),
    batch_size=272, seed=7,
    shuffle=False, class_mode='categorical'
)
for i in range(1):
    x1, y1 = valid_test.next()
    print(x1.shape, y1.shape)
    #print(y)
print(y1)
print(x1)
model = tf.keras.models.load_model('./mytrain.h5', compile=False)
model.summary()
y_pred = model.predict(x1)
#调用什么语句可以得到混淆矩阵？
y_pred = tf.argmax(y_pred, 1)
y1 = tf.argmax(y1, 1)
cm = confusion_matrix (y1, y_pred)
print(cm)