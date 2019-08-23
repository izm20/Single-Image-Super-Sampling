from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os import listdir
from os.path import isfile, join
from random import randint

import numpy as np
from skimage import data, io
from skimage.util import crop
from PIL import Image
from matplotlib import pyplot as plt

scale = 2
train_images = 150000
test_images = train_images + 200
epochs = 5
batch_size = 64
model_save_dir = os.getcwd() + '/model/'
output_dir = os.getcwd() + '/out/'

def denormalize(input_data):
    input_data = input_data * 255
    return input_data.astype(np.uint8)


def normalize(input_data):
    return input_data.astype(np.float32) / 255


# While training save generated image(in form LR, SR, HR)
# Save only one image as sample
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, x_test_bicubic, dim=(1, 4),
                          figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    value = randint(0, examples - 1)
    image_batch_hr = x_test_hr
    image_batch_lr = x_test_lr
    image_batch_bicubic = x_test_bicubic
    gen_img = generator.predict([image_batch_lr])
    gen_img = denormalize(gen_img)
    image_batch_hr = denormalize(image_batch_hr)
    image_batch_lr = denormalize(image_batch_lr)
    image_batch_bicubic = denormalize(image_batch_bicubic)

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(image_batch_bicubic[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(gen_img[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 4)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)
    print("Bicubic: " + str(psnr(image_batch_hr[value], image_batch_bicubic[value])))
    print("SR: " + str(psnr(image_batch_hr[value], gen_img[value])))


def psnr(target, ref):
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = np.math.sqrt(np.mean(diff ** 2.))

    return 20 * np.math.log10(255. / rmse)


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


def set_image_alignment(image, alignment):
    alignment = int(alignment)
    width, height = image.shape[1], image.shape[0]
    width = (width // alignment) * alignment
    height = (height // alignment) * alignment
    if image.shape[1] != width or image.shape[0] != height:
        image = image[:height, :width, :]

    return image


def get_split_images(image, window_size=48, stride=48):
    window_size = int(window_size)
    size = image.itemsize  # byte size of each value
    height, width = image.shape[0], image.shape[1]
    if stride is None:
        stride = window_size
    else:
        stride = int(stride)

    if height < window_size or width < window_size:
        return None

    new_height = 1 + (height - window_size) // stride
    new_width = 1 + (width - window_size) // stride

    shape = (new_height, new_width, window_size, window_size, 3)
    windows = np.zeros(shape).astype(np.uint8)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            a = i * stride
            b = window_size + a
            a1 = j * stride
            b1 = window_size + a1
            windows[i][j] = crop(image, ((a, image.shape[0] - b), (a1, image.shape[1] - b1), (0, 0)), copy=False)

    return windows


def images_loader(input_path):
    hr_images_train = []
    lr_images_train = []
    interpolated_images_train = []
    hr_images_test = []
    lr_images_test = []
    interpolated_images_test = []
    count = 0
    file_names = \
        [input_path + f for f in listdir(input_path) if (isfile(join(input_path, f)) and not f.startswith('.'))]
    for file in file_names:
        image = data.imread(file)
        image = set_image_alignment(image, scale)
        image_batch = get_split_images(image)
        image = resize_image(image, 1 / (scale * 2))
        image = set_image_alignment(image, scale)
        input_image = resize_image(image, 1 / scale)
        interpolated_image = resize_image(input_image, scale)
        hr_images_test.append(image)
        lr_images_test.append(input_image)
        interpolated_images_test.append(interpolated_image)

        # image = resize_image(image, 1 / scale)
        # image = set_image_alignment(image, scale)
        # input_image = resize_image(image, 1 / scale)
        # interpolated_image = resize_image(input_image, scale)
        # hr_images_train.append(image)
        # lr_images_train.append(input_image)
        # interpolated_images_train.append(interpolated_image)
        for i in range(0, image_batch.shape[0]):
            for j in range(0, image_batch.shape[1]):
                input_image = resize_image(image_batch[i][j], 1 / scale)
                interpolated_image = resize_image(input_image, scale)
                hr_images_train.append(image_batch[i][j])
                lr_images_train.append(input_image)
                interpolated_images_train.append(interpolated_image)
        count += 1
    return hr_images_test, lr_images_test, interpolated_images_test, hr_images_train, lr_images_train, interpolated_images_train