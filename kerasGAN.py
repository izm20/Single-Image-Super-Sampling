from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join
from random import randint

import numpy as np
from PIL import Image
from skimage import data, io
from matplotlib import pyplot as plt

import tensorflow as tf

import keras
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm

import Utils_model
from GAN import Generator, Discriminator
from Utils_model import VGG_LOSS

plt.switch_backend('agg')

print('TensorFlow version: ' + tf.__version__)
print('Keras version: ' + keras.__version__ + '\n')

scale = 2
train_images = 1000
test_images = 100
epochs = 10000
batch_size = 64
model_save_dir = os.getcwd() + '/model/'
output_dir = os.getcwd() + '/out/'

def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


# While training save generated image(in form LR, SR, HR)
# Save only one image as sample
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    print(examples)
    value = randint(0, examples - 1)
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)


def get_input_path():
    data_dir = os.getcwd() + '/data/train/'
    return data_dir


def resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    image = Image.fromarray(image, 'RGB')
    image = image.resize([width, height], resample=Image.BICUBIC)
    image = np.asarray(image)
    return image


def convert_rgb_to_y(image):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
    y_image = image.dot(xform.T) + 16.0

    return y_image


def images_loader(input_path):
    hr_images = []
    lr_images = []
    interpolated_images = []
    count = 0
    file_names = \
        [input_path + f for f in listdir(input_path) if (isfile(join(input_path, f)) and not f.startswith('.'))]
    for file in file_names:
        image = data.imread(file)
        image = resize_image(image, 1 / (scale * 2))
        input_image = resize_image(image, 1 / scale)
        interpolated_image = resize_image(image, scale)
        hr_images.append(image)
        lr_images.append(input_image)
        interpolated_images.append(interpolated_image)
        count += 1
    print("Read images: " + str(count))
    return hr_images, lr_images, interpolated_images


def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


if __name__ == "__main__":
    input_path = get_input_path()
    true_images, input_images, interpolated_images = images_loader(input_path)

    x_train_lr = np.array(input_images[:train_images])
    x_test_lr = np.array(input_images[train_images:(train_images + test_images)])

    x_train_hr = np.array(true_images[:train_images])
    x_test_hr = np.array(true_images[train_images:(train_images + test_images)])

    x_train_lr = normalize(x_train_lr)
    x_test_lr = normalize(x_test_lr)

    x_train_hr = normalize(x_train_hr)
    x_test_hr = normalize(x_test_hr)

    image_shape = true_images[0].shape
    shape = input_images[0].shape
    loss = VGG_LOSS(image_shape)
    batch_count = 2

    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    optimizer = Utils_model.get_optimizer()
    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)

    loss_file = open(model_save_dir + 'losses.txt', 'w+')
    loss_file.close()

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(batch_count)):
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_data_Y = np.random.random_sample(batch_size) * 0.2

            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])

        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)

        loss_file = open(model_save_dir + 'losses.txt', 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (
            e, gan_loss, discriminator_loss))
        loss_file.close()

        if e == 1 or e % 5 == 0:
            plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)
        if e % 500 == 0:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save(model_save_dir + 'dis_model%d.h5' % e)
