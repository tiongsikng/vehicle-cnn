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

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
with tf.device('/gpu:0'):
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

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.15)
    x_train, y_train = shuffle(x_train, y_train)

    seq = iaa.Sequential([
        sometimes(iaa.Affine(rotate=45)),
        sometimes(iaa.Crop(px=(0, 4)))
    ],random_order=True)

    seq2 = iaa.Sequential([
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
    ])

    x_train = seq.augment_images(x_train)
    # x_train = np.append(x_train, images_aug, axis=0)
    # y_train = np.append(y_train, y_train, axis=0)

    # images_aug2 = seq2.augment_images(x_train)
    # x_train = np.append(x_train, images_aug2, axis=0)
    # y_train = np.append(y_train, y_train, axis=0)


    x_train = x_train / 255.0
    x_test = x_test / 255.0


    pretrain_model = applications.VGG19(weights='imagenet',
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
    trainmodel = custom_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test,y_test))

    test_loss, test_acc = custom_model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    custom_model.save('vgg_model.h5')

    history = trainmodel.history
    history.keys()

    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    # epochs = range(1, len(acc) + 1)

    # # "bo" is for "blue dot"
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # # b is for "solid blue line"
    # plt.plot(epochs, val_loss, 'b', label='Testing loss')
    # # "bo" is for "blue dot"
    # plt.plot(epochs, acc, 'go', label='Training accuracy')
    # # b is for "solid blue line"
    # plt.plot(epochs, val_acc, 'g', label='Testing accuracy')
    # plt.title('Graph of training and testing loss/accuracy vs number of epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss/Accuracy')
    # plt.legend()

    # plt.show()
    # plt.savefig('evaluation.png')
