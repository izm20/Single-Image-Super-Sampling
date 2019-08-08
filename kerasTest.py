from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
from skimage import data, io
from matplotlib import pyplot as plt


import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils.data_utils import OrderedEnqueuer

print('TensorFlow version: ' + tf.__version__)
print('Keras version: ' + keras.__version__ + '\n')

scale = 2

def get_input_path():
    data_dir = os.getcwd() + '/data/train'
    if not data_dir.endswith('/'):
        path = data_dir + "/"
    return path
def resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    image = Image.fromarray(image, 'RGB')
    image = image.resize([width, height], resample=Image.BICUBIC)
    image = np.asarray(image)
    return image

def images_loader(input_path):
    hr_images = []
    lr_images = []
    interpolated_images = []
    count = 0
    file_names = \
        [input_path + f for f in listdir(input_path) if (isfile(join(input_path, f)) and not f.startswith('.'))]
    for file in file_names:
        image = data.imread(file)
        input_image = resize_image(image, 1 / scale)
        interpolated_image = resize_image(input_image, scale)
        hr_images.append(image)
        lr_images.append(input_image)
        interpolated_images.append(interpolated_image)
        count += count + 1
    return hr_images, lr_images, interpolated_images

if __name__ == "__main__":
    input_path = get_input_path()
    true_images, input_images, interpolated_images = images_loader(input_path)

