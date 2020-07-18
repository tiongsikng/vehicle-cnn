# from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.optimizers import SGD
from keras import applications
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Dense, Reshape, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from imgaug import augmenters as iaa
from keras.regularizers import l2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

img_rows, img_cols = 150, 150
batch_size = 64
epochs = 100
num_classes = 7
l2_reg=0.

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_cols, img_rows)
else:
    input_shape = (img_cols, img_rows, 3)
    
seq = iaa.Sequential([
    iaa.Crop(px=(0, 4)),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 0.5))
], random_order=True)

seq2 = iaa.Sequential([
    iaa.Crop(px=(0, 16)),
    iaa.GaussianBlur(sigma=(0, 0.75)),
    iaa.AdditiveGaussianNoise(scale=0.1*255)
], random_order=True)

train_datagen = ImageDataGenerator(rescale=None)

train_set = train_datagen.flow_from_directory(
'/media/videoDB/Train',
classes=['Bus','Car','Motorcycle','Trailer','Road','PickUp','Taxi'],
target_size = (img_rows, img_cols),
batch_size = batch_size,
class_mode = 'categorical')

(x_train, y_train) = train_set.next()
(x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.20)

images_aug = seq.augment_images(x_train)
new_images_aug = seq2.augment_images(x_train)
new_x_train = np.append(x_train, images_aug, axis=0)
new_y_train = np.append(y_train, y_train, axis=0)

new_x_train = np.append(new_x_train, new_images_aug, axis=0)
new_y_train = np.append(new_y_train, y_train, axis=0)

new_x_train = new_x_train / 255.0
x_test = x_test / 255.0

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

vgg_model = applications.VGG19(weights='imagenet',
                               include_top=False,
                               input_shape=input_shape)

x = vgg_model.output

# Stacking a new simple convolutional network on top of it   
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(x)
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
#x = AveragePooling2D(pool_size=(2, 2))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
#x = AveragePooling2D(pool_size=(2, 2))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
# x = AveragePooling2D(pool_size=(2, 2))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)

# # x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
# # x = Conv2D(filters=256, kernel_size=(1, 1), activation='relu', padding='same')(x)
# # # x = AveragePooling2D(pool_size=(2, 2))(x)
# # x = Dropout(0.5)(x)
# # x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

# Creating new model. Please note that this is NOT a Sequential() model.
from keras.models import Model
custom_model = Model(input=vgg_model.input, output=x)

# Make sure that the pre-trained bottom layers are not trainable
for layer in vgg_model.layers:
        layer.trainable = False


# Do not forget to compile it
custom_model.compile(loss='categorical_crossentropy',
                     optimizer='sgd',
                     metrics=['accuracy'])
                     
custom_model.summary()
custom_model.fit(new_x_train, new_y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
