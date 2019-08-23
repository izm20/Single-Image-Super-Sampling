from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir
from os.path import isfile, join
from random import randint

import numpy as np
from PIL import Image
from keras.layers import subtract, K
from keras.optimizers import Adam
from keras.utils import plot_model
from matplotlib import pyplot as plt
from skimage import data, io
from skimage.util import crop


def check_path(path):
    if not path.endswith('/'):
        path = path + "/"
    return path


def denormalize(input_data):
    input_data = input_data * 255
    return input_data.astype(np.uint8)


def normalize(input_data):
    return input_data.astype(np.float32) / 255


def psnr(target, ref):
    target_data = np.array(target, dtype=float)
    ref_data = np.array(ref, dtype=float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = np.math.sqrt(np.mean(diff ** 2.))
    return 20 * np.math.log10(255. / rmse)


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


def build_image_from_batch(image_batch, batch_shape):
    stride_height = image_batch.shape[1]
    stride_width = image_batch.shape[2]
    height = int(batch_shape[0] * stride_height)
    width = int(batch_shape[1] * stride_width)
    channels = image_batch.shape[3]
    image = np.zeros(shape=(height, width, channels)).astype(np.float32)
    index = 0
    for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
            height_start = stride_height * i
            height_end = stride_height * (i + 1)
            width_start = stride_width * j
            width_end = stride_width * (j + 1)
            image[height_start:height_end, width_start:width_end, :channels] = image_batch[index]
            index += 1
    image = denormalize(image)
    image = resize_image(image, 1 / 2)
    image = normalize(image)
    return image


def images_loader(input_path, scale):
    hr_images_train = []
    lr_images_train = []
    hr_images_test = []
    lr_images_test = []
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
        hr_images_test.append(image)
        lr_images_test.append(input_image)
        for i in range(0, image_batch.shape[0]):
            for j in range(0, image_batch.shape[1]):
                input_image = resize_image(image_batch[i][j], 1 / scale)
                hr_images_train.append(image_batch[i][j])
                lr_images_train.append(input_image)
                count += 1
    print(str(count) + ' images loaded')
    return hr_images_test, lr_images_test, hr_images_train, lr_images_train


def split_for_test(image):
    image_batch = get_split_images(image)
    batch_shape = image_batch.shape
    image_array = []
    for i in range(0, image_batch.shape[0]):
        for j in range(0, image_batch.shape[1]):
            image_array.append(image_batch[i][j])
    image_batch = np.array(image_array)
    return image_batch, batch_shape


def plot_generated_test(output_dir, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    random_index = randint(0, examples - 1)
    image_hr = x_test_hr[random_index]
    image_input = np.array(x_test_lr[random_index])
    image_input = image_input.reshape((1, image_input.shape[0], image_input.shape[1], image_input.shape[2]))
    image_generated = generator.predict([image_input])
    image_generated = image_generated.reshape(
        (image_generated.shape[1], image_generated.shape[2], image_generated.shape[3]))
    image_input = image_input.reshape(
        (image_input.shape[1], image_input.shape[2], image_input.shape[3]))
    image_generated = denormalize(image_generated)
    image_hr = denormalize(image_hr)
    image_input = denormalize(image_input)
    plt.figure(figsize=figsize)
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_input, interpolation='nearest')
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(image_generated, interpolation='nearest')
    plt.axis('off')
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_hr, interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_test.png')
    print("PSNR for image generated: " + str(psnr(image_hr, image_generated)))


def save_image(path, image):
    path = path + 'generated.png'
    io.imsave(path, image)
    print('Image: ' + path + ' saved')


def get_optimizer():
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam


def content_loss(hr, sr):
    diff = subtract([sr, hr])
    mse = K.mean(K.square(diff))
    loss = K.identity(mse)
    return loss


def save_model(model, loss_history, path):
    plot_model(model, to_file=path + 'model.png')
    print('Model structure saved: ' + path + 'model.png')
    loss_history = str(loss_history.history['loss'])
    loss_file = open(path + 'losses.txt', 'w+')
    loss_file.write('loss = ' + loss_history + ';\n')
    loss_file.close()
    print('Losses history saved: ' + path + 'losses.txt')
    model.save(path + 'model_saved.h5')
    print('Model saved: ' + path + 'model_saved.h5')
