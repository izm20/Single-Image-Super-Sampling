from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm


print('TensorFlow version: ' + tf.__version__)
print('Keras version: ' + keras.__version__ + '\n')

EPOCHS = 30
BATCH_SIZE = 100
LOW_IMG_SHAPE =  [240, 426, 3]
HIGH_IMG_SHAPE = [480, 854, 3]

BASE_DIR = os.getcwd()
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
TRAIN_DIR = os.path.join(IMAGE_DIR, 'train')
VAL_DIR = os.path.join(IMAGE_DIR, 'val')
TEST_DIR = os.path.join(IMAGE_DIR, 'test')


NUM_TRAIN = len(os.listdir(TRAIN_DIR))
NUM_VAL = len(os.listdir(VAL_DIR))
NUM_TEST = len(os.listdir(TEST_DIR))

print('Total training images: ', NUM_TRAIN)
print('Total validation images: ', NUM_VAL)
print('Total test images: ', NUM_TEST)

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE, directory=TRAIN_DIR, shuffle=True, target_size= (LOW_IMG_SHAPE[0], LOW_IMG_SHAPE[1]), class_mode='binary')

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE, directory=VAL_DIR, shuffle=False, target_size= (LOW_IMG_SHAPE[0], LOW_IMG_SHAPE[1]), class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE, directory=TEST_DIR, shuffle=False, target_size= (LOW_IMG_SHAPE[0], LOW_IMG_SHAPE[1]), class_mode='binary')


sample_training_images, _ = next(train_data_gen)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 2, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:2])
