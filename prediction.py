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
import time


start = time.time()
img_rows, img_cols = 150, 150
num_classes = 7
class_names = ['5Ton','BinTruck','BoxTruck','Bus','Car','Carrier','Cement','Crane','Motorcycle','OilTanker','OpenBackTruck','PickUp','Road','Taxi','Tow','TowFlat','Tractor','Trailer','TrashTruck']

# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_cols, img_rows)
# else:
#     input_shape = (img_cols, img_rows, 3)

# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('vgg_model.h5')
new_model.summary()

x_test = joblib.load('x_test.pkl')
y_test = joblib.load('y_test.pkl')

x_test = x_test / 255.0

loss, acc = new_model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

class_nm = '5Ton'
print('Predictions for class', class_nm)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm1 = 'BinTruck'
print('Predictions for class', class_nm1)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm1 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm2 = 'BoxTruck'
print('Predictions for class', class_nm2)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm2 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm3 = 'Bus'
print('Predictions for class', class_nm3)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm3 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm4 = 'Car'
print('Predictions for class', class_nm4)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm4 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm5 = 'Carrier'
print('Predictions for class', class_nm5)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm5 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm6 = 'Cement'
print('Predictions for class', class_nm6)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm6 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm7 = 'Crane'
print('Predictions for class', class_nm7)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm7 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm8 = 'Motorcycle'
print('Predictions for class', class_nm8)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm8 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm9 = 'OilTanker'
print('Predictions for class', class_nm9)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm9 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm10 = 'OpenBackTruck'
print('Predictions for class', class_nm10)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm10 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm11 = 'PickUp'
print('Predictions for class', class_nm11)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm11 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm12 = 'Road'
print('Predictions for class', class_nm12)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm12 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm13 = 'Taxi'
print('Predictions for class', class_nm13)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm13 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm14 = 'Tow'
print('Predictions for class', class_nm14)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm14 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm15 = 'TowFlat'
print('Predictions for class', class_nm15)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm15 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm16 = 'Tractor'
print('Predictions for class', class_nm16)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm16 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm17 = 'Trailer'
print('Predictions for class', class_nm17)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm17 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
print("\n")

class_nm18 = 'TrashTruck'
print('Predictions for class', class_nm18)
for i in range (10):
    test_image = image.load_img('/media/videoDB/tsng/Test/' + class_nm18 + '/' + str(i + 1) + '.png', target_size = (150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    predictions = new_model.predict(test_image)
    # print(np.argmax(predictions))
    print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
    
end = time.time()
print(end - start, "seconds")

# def prepare(filepath):
#     IMG_SIZE = 150  # 50 in txt-based
#     img_array = cv2.imread(filepath)  # read in the image, convert to grayscale
#     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
#     return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.

# class_nm = '5Ton'
# print('Prediction for class: ', class_nm)
# for i in range (10):
#     prediction = new_model.predict([prepare('/media/videoDB/tsng/Test/' + class_nm + '/' + str(i + 1) + '.png')])
#     print(class_names[int(prediction[0][0])])

