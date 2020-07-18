from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from imgaug import augmenters as iaa
from keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from tensorflow import set_random_seed
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import pickle
from sklearn.externals import joblib

# np.random.seed(33)
# random.seed(33)
# set_random_seed(33)

img_rows, img_cols = 150, 150
num_classes = 19
training_data = []
test_data = []
TEST_DATADIR = '/media/videoDB/tsng/Test'
TRAIN_DATADIR = '/media/videoDB/tsng/Train1'
class_names = ['5Ton','BinTruck','BoxTruck','Bus','Car','Carrier','Cement','Crane','Motorcycle','OilTanker','OpenBackTruck','PickUp','Road','Taxi','Tow','TowFlat','Tractor','Trailer','TrashTruck']
x_train = []
y_train = []
x_test = []
y_test = []

# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_cols, img_rows)
# else:
#     input_shape = (img_cols, img_rows, 3)

with tf.device('/gpu:0'):
    def create_training_data():
        for category in class_names:
            path = os.path.join(TRAIN_DATADIR,category)
            class_num = class_names.index(category)

            for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
                try:
                    img_array = cv2.imread(os.path.join(path,img))  # convert to array
                    new_array = cv2.resize(img_array, (img_cols, img_rows))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e: 
                    pass
        return training_data
    
    training_data = create_training_data()
    
    # random.shuffle(training_data)
    for features,label in training_data:
        x_train.append(features)
        y_train.append(label)
    x_train = np.array(x_train).reshape(len(training_data), img_rows, img_cols, 3)
    y_train = np.array(y_train).reshape(len(training_data))

    joblib.dump(x_train, "x_train.pkl")
    joblib.dump(y_train, "y_train.pkl")
    
    def create_test_data():
        for category in class_names:
            path = os.path.join(TEST_DATADIR,category)
            class_num = class_names.index(category) 

            for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
                try:
                    img_array = cv2.imread(os.path.join(path,img))  # convert to array
                    new_array = cv2.resize(img_array, (img_cols, img_rows))  # resize to normalize data size
                    test_data.append([new_array, class_num])  # add this to our test_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
        return test_data
    
    test_data = create_test_data()
    
    for features,label in test_data:
        x_test.append(features)
        y_test.append(label)
    x_test = np.array(x_test).reshape(len(test_data), img_rows, img_cols, 3)
    y_test = np.array(y_test).reshape(len(test_data))

    joblib.dump(x_test, "x_test.pkl")
    joblib.dump(y_test, "y_test.pkl")
