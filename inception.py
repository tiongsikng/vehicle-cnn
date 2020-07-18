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

img_rows, img_cols = 150, 150
batch_size = 32
epochs = 100
num_classes = 19
l2_reg=0.
class_names = ['5Ton','BinTruck','BoxTruck','Bus','Car','Carrier','Cement','Crane','Motorcycle','OilTanker','OpenBackTruck','PickUp','Road','Taxi','Tow','TowFlat','Tractor','Trailer','TrashTruck']

x_train = joblib.load('x_train.pkl')
y_train = joblib.load('y_train.pkl')
# x_test = joblib.load('x_test.pkl')
# y_test = joblib.load('y_test.pkl')

(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.15)

seq = iaa.Sequential([
    iaa.Crop(px=(0, 4)),
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=45)
])

seq2 = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
])

# images_aug = seq.augment_images(x_test)
# x_test = np.append(x_test, images_aug, axis=0)
# y_test = np.append(y_test, y_test, axis=0)

# images_aug2 = seq2.augment_images(x_test)
# x_test = np.append(x_test, images_aug2, axis=0)
# y_test = np.append(y_test, y_test, axis=0)

x_train = x_train / 255.0
x_test = x_test / 255.0


pretrain_model = applications.InceptionV3(weights='imagenet',
                              include_top=False,
                              input_shape=x_train.shape[1:])

x = pretrain_model.output

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

custom_model = Model(input=pretrain_model.input, output=x)

# Make sure that the pre-trained bottom layers are not trainable
for layer in pretrain_model.layers:
        layer.trainable = False


# Do not forget to compile it
custom_model.compile(loss='sparse_categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
                     
custom_model.summary()
custom_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test,y_test))

custom_model.save('inception_model.h5')
