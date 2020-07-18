from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from imgaug import augmenters as iaa
from keras.regularizers import l2
import numpy as np
from tensorflow import set_random_seed
import random
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('vgg_model.h5')
new_model.summary()

x_train = joblib.load('x_train.pkl')
y_train = joblib.load('y_train.pkl')

(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.15)

x_train = x_train / 255.0
x_test = x_test / 255.0

loss, acc = new_model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
