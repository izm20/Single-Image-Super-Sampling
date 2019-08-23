from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join
from random import randint

import numpy as np

import tensorflow as tf

import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import subtract
from keras.utils import plot_model
from keras.models import load_model
from skimage import io, data

from DeepCNN import Generator_Model
from Utils import get_input_path, normalize, plot_generated_images, images_loader, resize_image, set_image_alignment, \
    denormalize, psnr

print('TensorFlow version: ' + tf.__version__)
print('Keras version: ' + keras.__version__ + '\n')

scale = 2
train_images = 150000
test_images = train_images + 200
epochs = 5
batch_size = 64
model_save_dir = os.getcwd() + '/model/'
output_dir = os.getcwd() + '/out/'


def get_optimizer():
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam


def content_loss(hr, sr):
    diff = subtract([sr, hr])
    mse = K.mean(K.square(diff))
    loss = K.identity(mse)
    return loss


# if __name__ == "__main__":
#     input_path = get_input_path()
#     hr_images_test, lr_images_test, interpolated_images_test, hr_images_train, lr_images_train, interpolated_images_train = images_loader(
#         input_path)
#
#     # x_train_hr = np.array(hr_images_test[:train_images])
#     # x_train_lr = np.array(lr_images_test[:train_images])
#     # x_train_bicubic = np.array(interpolated_images_test[:train_images])
#
#     x_train_hr = np.array(hr_images_train)
#     x_train_lr = np.array(lr_images_train)
#     x_train_bicubic = np.array(interpolated_images_train)
#
#     # x_test_hr = np.array(hr_images_test[train_images:test_images])
#     # x_test_lr = np.array(lr_images_test[train_images:test_images])
#     # x_test_bicubic = np.array(interpolated_images_test[train_images:test_images])
#
#     x_test_hr = np.array(hr_images_test)
#     x_test_lr = np.array(lr_images_test)
#     x_test_bicubic = np.array(interpolated_images_test)
#
#     x_train_hr = normalize(x_train_hr)
#     x_test_hr = normalize(x_test_hr)
#     x_train_lr = normalize(x_train_lr)
#     x_test_lr = normalize(x_test_lr)
#     x_train_bicubic = normalize(x_train_bicubic)
#     x_test_bicubic = normalize(x_test_bicubic)
#
#     image_shape = hr_images_test[0].shape
#     shape = lr_images_test[0].shape
#
#     generator = Generator_Model(shape, image_shape).generator()
#     # generator = Generator_Model(shape, image_shape).srResNet()
#     optimizer = get_optimizer()
#     generator.compile(loss=content_loss, optimizer=optimizer, metrics=['accuracy'])
#     loss_file = open(model_save_dir + 'losses.txt', 'w+')
#     loss_file.close()
#     plot_model(generator, to_file=model_save_dir + 'model.png')
#     for e in range(1):
#         loss_history = generator.fit([x_train_lr], x_train_hr, batch_size, epochs, 1)
#
#         loss_history = str(loss_history.history['loss'])
#         loss_file = open(model_save_dir + 'losses.txt', 'a')
#         loss_file.write('epoch%d : loss = %s ;\n' % (e, loss_history))
#         loss_file.close()
#
#         generator.save(model_save_dir + 'gen_model%d.h5' % e)
#         plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr, x_test_bicubic)

if __name__ == "__main__":
    input_path = get_input_path()
    model = load_model('./model/srcnn_model.h5', custom_objects={'content_loss': content_loss})
    file_names = \
        [input_path + f for f in listdir(input_path) if (isfile(join(input_path, f)) and not f.startswith('.'))]
    image = data.imread('./images/HR/frames_1/frame122.jpg')
    # image = resize_image(image, 1 / scale)
    image = set_image_alignment(image, scale)
    image_lr = resize_image(image, 1 / scale)
    image_bicubic = resize_image(image_lr, scale)
    image_lr = image_lr.reshape((1, image_lr.shape[0], image_lr.shape[1], image_lr.shape[2]))
    image_bicubic = image_bicubic.reshape((1, image_bicubic.shape[0], image_bicubic.shape[1], image_bicubic.shape[2]))
    image_lr = normalize(image_lr)
    image_bicubic = normalize(image_bicubic)
    generated = model.predict([image_lr])
    generated = np.array(generated)
    generated = generated.reshape((generated.shape[1], generated.shape[2], generated.shape[3]))
    image_lr = image_lr.reshape((image_lr.shape[1], image_lr.shape[2], image_lr.shape[3]))
    image_bicubic = image_bicubic.reshape((image_bicubic.shape[1], image_bicubic.shape[2], image_bicubic.shape[3]))
    image_lr = denormalize(image_lr)
    image_bicubic = denormalize(image_bicubic)
    generated = denormalize(generated)
    io.imsave('./out/generated.png', generated)
    io.imsave('./out/bicubic.png', image_bicubic)
    io.imsave('./out/image_lr.png', image_lr)
    io.imsave('./out/image.png', image)
    print('Generated: ' + str(psnr(image, generated)))
    print('Bicubic: ' + str(psnr(image, image_bicubic)))

