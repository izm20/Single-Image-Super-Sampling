import argparse
from os.path import isfile

import keras
import tensorflow as tf
from keras.engine.saving import load_model
from skimage import data

from Utils import content_loss, set_image_alignment, normalize, denormalize, save_image, \
    build_image_from_batch, split_for_test, check_path

HEIGHT_LIMITED = 900
WIDTH_LIMITED = 1500


class Test(object):

    def __init__(self, input_path, output_path, model_path):
        self.input_path = input_path
        self.model_path = model_path
        self.output_path = check_path(output_path)

    def test(self):
        if self.model_path[len(self.model_path) - 3:] != '.h5':
            print('Error: incompatible model file')
            exit()
        model = load_model(filepath=self.model_path, custom_objects={'content_loss': content_loss})

        if isfile(self.input_path):
            image = data.imread(self.input_path)
            image = set_image_alignment(image, 2)
            if image.shape[0] >= HEIGHT_LIMITED or image.shape[1] >= WIDTH_LIMITED:
                image_batch, batch_shape = split_for_test(image)
                image_batch = normalize(image_batch)
                image_generated_batch = model.predict([image_batch])
                image_generated = build_image_from_batch(image_generated_batch, batch_shape)
            else:
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                image = normalize(image)
                image_generated = model.predict([image])
                image_generated = image_generated.reshape(
                    (image_generated.shape[1], image_generated.shape[2], image_generated.shape[3]))

            image = denormalize(image_generated)
            save_image(self.output_path, image)


if __name__ == "__main__":
    print('TensorFlow version: ' + tf.__version__)
    print('Keras version: ' + keras.__version__ + '\n')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./data/input.png',
                        help='Path for input images')

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./out/',
                        help='Path for Output images')

    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/model.h5',
                        help='Path for model')

    args = parser.parse_args()

    test = Test(args.input_dir, args.output_dir, args.model_dir)
    test.test()
