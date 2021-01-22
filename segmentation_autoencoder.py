import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dropout,Conv2D,MaxPooling2D,LeakyReLU,UpSampling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import os
import cv2
import matplotlib.image as mpimg
import math
from sklearn.model_selection import train_test_split


training_images = []
segmented_images = [] 
folders = []
# append folders containing original and segmented images 
folders.append('image_folder_path')  
folders.append('image_folder_path')

for each_folder in folders: 
  for filename in os.listdir(each_folder):
    if filename.rpartition('.')[2] == 'jpg': # from the . onwards what is the file extension
      img = mpimg.imread(os.path.join(each_folder,filename)) # get original image
      filename_seg = filename.rpartition('.')[0] + '_seg.png' # find the segmented version of the image
      img_seg = mpimg.imread(os.path.join(each_folder,filename_seg))
      training_images.append(img) # create list with images
      segmented_images.append(img_seg) # create list with segmented images, with preserved pair order
    else:
      continue

tensor_shape = (-1,) + np.shape(training_images[0])
training_images = np.reshape(training_images, (tensor_shape))/ 255. # reshape to tensors and normalize
segmented_images = np.reshape(segmented_images, (tensor_shape))/ 255. # reshape to tensors and normalize

#%%
percent_train = 0.9 # percentage of all the data that will go to train 
# create shuffle order, so training and segmentation images are shuffled the same way to preserve pairs
new_order = np.random.permutation(np.shape(training_images)[0])
t = [training_images[i] for i in new_order] # shuffle the training images based on new order
s = [segmented_images[ii] for ii in new_order] # shuffle the training images based on new order

all_idx = list(range(len(t)))
train_idx, test_idx = train_test_split(all_idx,test_size=(1-percent_train)) # split into train and test
# convert train and test images from lists back to arrays
x_train = np.asarray([t[iii] for iii in train_idx])
x_test = np.asarray([t[iii] for iii in test_idx])
x_train_seg = np.asarray([s[iv] for iv in train_idx])
x_test_seg = np.asarray([s[iv] for iv in test_idx])

#%%
# =============================== Create Model ================================

input_img = Input(shape=np.shape(x_test[0]))

conv1 = 16
conv2 = 32
conv3 = 64
conv4 = 128
dropout_rate = 0.1
stride_val = (1,1)

kernel_size = (3,3) 

x = Conv2D(conv1, kernel_size, activation='relu', strides=stride_val, padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(conv2, kernel_size, activation='relu', strides=stride_val, padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(conv3, kernel_size, activation='relu', strides=stride_val, padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(dropout_rate)(x)
x = Conv2D(conv4, kernel_size, activation='relu', strides=stride_val, padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(conv4, kernel_size, activation='relu', strides=stride_val, padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(conv3, kernel_size, activation='relu', strides=stride_val, padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(conv2, kernel_size, activation='relu', strides=stride_val, padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(conv1, kernel_size, activation='relu', strides=stride_val, padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, kernel_size, activation='sigmoid', strides=stride_val, padding='same')(x)

autoencoder = Model(input_img, decoded)
opt=keras.optimizers.Adam(lr=1e-3)
autoencoder.compile(opt, loss='binary_crossentropy')

autoencoder.summary()

# ================================ Train Model ================================

history = autoencoder.fit(x_train, x_train_seg,
                          epochs=800,
                          batch_size = 32,
                          shuffle = True,
                          validation_data=(x_test, x_test_seg))

# visualize loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#%%
# ================================ Save Model =================================
from keras.models import model_from_json
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
autoencoder.save_weights("autoencoder.h5")















