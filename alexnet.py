# Import necessary components to build AlexNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from imgaug import augmenters as iaa
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
import os
import cv2
import random

img_rows, img_cols = 150, 150
batch_size = 32
epochs = 100
num_classes = 19
l2_reg=0.

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# split data and normalize the pixels
(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.15)
x_train, x_test = x_train / 255.0, x_test/ 255.0

# Initialize model
alexnet = Sequential()

# Layer 1, kernel_regularizer=l2(l2_reg)
alexnet.add(Conv2D(96, (11, 11), input_shape=x_train.shape[1:],
	padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.8))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
alexnet.add(Conv2D(256, (5, 5), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.8))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(512, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.8))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.8))

# Layer 5
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.8))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
alexnet.add(Flatten())
alexnet.add(Dense(3072))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.8))

# Layer 7
alexnet.add(Dense(4096))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.8))

# Layer 8
alexnet.add(Dense(num_classes))
alexnet.add(BatchNormalization())
alexnet.add(Activation('softmax'))

# compile model
adam = optimizers.Adam(lr=0.001)
alexnet.compile(optimizer=adam, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
# view model summary and start training
alexnet.summary()
alexnet.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

alexnet.save('alexnet_model.h5')
