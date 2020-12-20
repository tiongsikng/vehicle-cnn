from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from imgaug import augmenters as iaa
from keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from tensorflow import set_random_seed
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from tqdm import tqdm

# np.random.seed(33)
# random.seed(33)
# set_random_seed(33)

img_rows, img_cols = 150, 150

TEST_DATADIR = '/home/train'
TRAIN_DATADIR = '/home/test'
class_names = ['5Ton','BinTruck','BoxTruck','Bus','Car','Carrier','Cement','Crane','Motorcycle','OilTanker','OpenBackTruck','PickUp','Road','Taxi','Tow','TowFlat','Tractor','Trailer','TrashTruck']
num_classes = len(class_names)

train_data = []
test_data = []
x_train = []
y_train = []
x_test = []
y_test = []

# handle image format between PIL and OpenCV
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_cols, img_rows)
else:
    input_shape = (img_cols, img_rows, 3)


def create_train_data():
    for category in class_names:
        path = os.path.join(TRAIN_DATADIR,category)
        class_num = class_names.index(category)

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (img_cols, img_rows))  # resize to normalize data size
                train_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e: 
                pass

    return train_data

def create_test_data():
    for category in class_names:
        path = os.path.join(TEST_DATADIR,category)
        class_num = class_names.index(category) 

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (img_cols, img_rows))  # resize to normalize data size
                test_data.append([new_array, class_num])  # add this to our test_data
            except Exception as e:
                pass

    return test_data        

def get_array(data_array):
    for features,label in data_array:
        x.append(features)
        y.append(label)
    x = np.array(x).reshape(len(test_data), img_rows, img_cols, 3)
    y = np.array(y).reshape(len(test_data))

    return x, y

if __name__ == '__main__':
    with tf.device('/gpu:0'):
        train_data = create_train_data()
        x_train, y_train = get_array(train_data)
        test_data = create_test_data()
        x_test, y_test = get_array(test_data)

        # save data as numpy array
        np.save('x_train.npy', x_train)
        np.save('y_train.npy', y_train)
        np.save('x_test.npy', x_test)
        np.save('y_test.npy', y_test)
