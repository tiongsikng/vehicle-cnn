from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow import keras
from keras.models import Sequential
from keras import applications
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Dense, Reshape, Activation
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from imgaug import augmenters as iaa
from keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from tensorflow import set_random_seed
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.utils import shuffle
from keras.utils import plot_model
import matplotlib.pyplot as plt

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

with tf.device('/gpu:0'):
    img_rows, img_cols = 150, 150
    batch_size = 32
    epochs = 100
    num_classes = 19
    l2_reg=0.
    class_names = ['5Ton','BinTruck','BoxTruck','Bus','Car','Carrier','Cement','Crane','Motorcycle','OilTanker','OpenBackTruck','PickUp','Road','Taxi','Tow','TowFlat','Tractor','Trailer','TrashTruck']

    x_train = joblib.load('x_train.pkl')
    y_train = joblib.load('y_train.pkl')

    (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.15)

    #define model
    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=l2(l2_reg)),
        # keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.AveragePooling2D(pool_size=2, strides=2),
        keras.layers.Dropout(0.5),   
    
    # keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.BatchNormalization(),
    # keras.layers.AveragePooling2D(pool_size=2),
    # keras.layers.Dropout(0.5),   
    
    # keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.BatchNormalization(),
    # keras.layers.AveragePooling2D(pool_size=2),
    # keras.layers.Dropout(0.5),   
    
    # keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    # keras.layers.AveragePooling2D(pool_size=2),
    # keras.layers.BatchNormalization(),
    # keras.layers.Dropout(0.5),   
    
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),    
        keras.layers.BatchNormalization(),
        keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    trainmodel = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=batch_size)

    model.save('model.h5')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    
    history = trainmodel.history
    history.keys()

    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Testing loss')
    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'go', label='Training accuracy')
    # b is for "solid blue line"
    plt.plot(epochs, val_acc, 'g', label='Testing accuracy')
    plt.title('Graph of training and testing loss/accuracy vs number of epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()

    plt.show()
    plt.savefig('model.png')

