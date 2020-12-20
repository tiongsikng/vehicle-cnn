from keras import applications
from keras.preprocessing.image import ImageDataGenerator
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

img_rows, img_cols = 150, 150
class_names = ['5Ton','BinTruck','BoxTruck','Bus','Car','Carrier','Cement','Crane','Motorcycle','OilTanker','OpenBackTruck','PickUp','Road','Taxi','Tow','TowFlat','Tractor','Trailer','TrashTruck']
num_classes = len(class_names)
true_predict = 0
false_predict = 0

model = keras.models.load_model('vgg_model.h5')
model.summary()

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
x_test = x_test / 255.0

loss, acc = new_model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# predict image array by means of loading the data from directory (ground truth)
for i in range(len(class_names)):
    print('Predictions for class', class_names[i])
    for j in range(10):
        test_image = image.load_img('/home/Test/' + class_names[i] + '/' + str(j + 1) + '.png', target_size = (150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        predictions = model.predict(test_image)
        # print(np.argmax(predictions))
        print(np.argmax(predictions), class_names[np.argmax(predictions, axis=1)[0]])
        if class_names[np.argmax(predictions, axis=1)[0]] == class_names[i]:
            true_predict +=1
        else:
            false_predict += 1   
    print("****\n")
predict_acc = (true_predict / (true_predict + false_predict)) * 100
print("Prediction accuracy: %.2f" %(predict_acc))


# calculate model prediction from x_test by comparing with ground truth in y_test
predictions = model.predict(x_test)
test_image = np.expand_dims(x_test, axis = 0) 
arr = model.predict(test_image[0])

for i in range (0, x_test.shape[0]):
  if y_test[i] == np.argmax(arr[i]):
    true_predict += 1
  else:
    false_predict += 1
predict_acc = (true_predict / (true_predict + false_predict)) * 100
print("Prediction accuracy: %.2f" %(predict_acc))