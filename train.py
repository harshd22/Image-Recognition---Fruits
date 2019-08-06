#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPool2D,Conv2D, Flatten, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
import time
from imutils import paths
import pandas as pd

import matplotlib.pyplot as matplot
import numpy as np
import tensorflow as tf
import cv2
import os
import random
import argparse

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


def load_images(train_data_dir, image_size = (300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    images_data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(train_data_dir)))
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        images_data.append(image)


        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    return images_data, sorted(labels)

def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    X_train = np.array(images, dtype = "float")
    y_train = np.array(labels)


    return X_train, y_train

def construct_model():
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten(input_shape=(64,64,3)))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_logarithmic_error',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    return model


def train_model(model,X_train,Y_train):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
    #  do constant normalize
    X_train = X_train / 255.0
    # Binarize the labels
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    #split test and train sets
    X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train, test_size=0.2)
    # Image Augmentation
    datagen = ImageDataGenerator(
        rotation_range=45,
        height_shift_range=10,
        horizontal_flip=True,
        zoom_range=0.1,
        width_shift_range=10
    )
    datagen.fit(X_train)
    time1 = time.time()
    history = model.fit_generator(datagen.flow(X_train,Y_train,batch_size=25 ),
                        epochs = 32,
                        steps_per_epoch=len(X_train)/25,
                        verbose=2,
                       validation_data=(X_test,Y_test),

                        )
    time2 = time.time()

    timeTaken = time2-time1
    print("Time taken to train the model: (in minutes)" )
    print(timeTaken/60)
    plotGraph(history)
    return model

def plotGraph(history):

    matplot.plot(history.history['acc'])
    matplot.plot(history.history['val_acc'])
    matplot.title('Model Accuracy')
    matplot.ylabel('Accuracy')
    matplot.xlabel('epochs')
    matplot.legend(['Train','Test'], loc='upper right')
    matplot.show()


    matplot.plot(history.history['loss'])
    matplot.plot(history.history['val_loss'])
    matplot.title('Model Loss')
    matplot.ylabel('Loss')
    matplot.xlabel('epochs')
    matplot.legend(['Train', 'Test'], loc='upper right')
    matplot.show()


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***

    model.save("model/model.h5")
    print("Model Saved Successfully.")


if __name__ == '__main__':


    # Image size, please define according to your settings when training your model.
    image_size = (64, 64)

    # Load images
    images, labels = load_images("Train_data/Train_data", image_size)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    X_train, y_train = convert_img_to_array(images, labels)


    model = construct_model()
    model = train_model(model,X_train,y_train)
    save_model(model)

  
