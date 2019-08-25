import argparse
import datetime

import keras
import numpy as np
import tensorflow as tf
from keras import metrics

from SRDeepCNN import SRDeepCNN
from Utils import content_loss, normalize, check_path, images_loader, plot_generated_test, get_optimizer, save_model, \
    images_loader_mini


class Train(object):

    def __init__(self, input_dir, output_dir, model_dir, epochs, batch_size, scale, channels=3):
        self.input_dir = check_path(input_dir)
        self.output_dir = check_path(output_dir)
        self.model_dir = check_path(model_dir)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.scale = int(scale)
        self.channels = int(channels)

    def train(self):
        logdir = self.model_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        hr_images_test, lr_images_test, hr_images_train, lr_images_train = images_loader_mini(self.input_dir, self.scale)
        y_train_hr = np.array(hr_images_train[:500000])
        x_train_lr = np.array(lr_images_train[:500000])
        y_test_hr = np.array(hr_images_test[:15000])
        x_test_lr = np.array(lr_images_test[:15000])
        y_train_hr = normalize(y_train_hr)
        y_test_hr = normalize(y_test_hr)
        x_train_lr = normalize(x_train_lr)
        x_test_lr = normalize(x_test_lr)
        model = SRDeepCNN(self.channels, self.scale).build_model()
        model.compile(loss=content_loss, optimizer=get_optimizer(), metrics=[metrics.mse, metrics.categorical_accuracy])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, batch_size=self.batch_size, write_graph=True, write_images=True, write_grads=True)
        loss_history = model.fit(x_train_lr, y_train_hr, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                                 validation_data=([x_test_lr, y_test_hr]), callbacks=[tensorboard_callback])
        save_model(model, loss_history, self.model_dir)
        plot_generated_test(self.output_dir, model, y_test_hr, x_test_lr)


if __name__ == "__main__":
    print('TensorFlow version: ' + tf.__version__)
    print('Keras version: ' + keras.__version__ + '\n')

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./data/',
                        help='Path for input images')

    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./out/',
                        help='Path for Output images')

    parser.add_argument('-m', '--model_dir', action='store', dest='model_dir', default='./model/',
                        help='Path for model')

    parser.add_argument('-e', '--epochs', action='store', dest='epochs', default='5',
                        help='Number of epochs for train')

    parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default='64',
                        help='Batch size for train')

    parser.add_argument('-s', '--scale', action='store', dest='scale', default='2',
                        help='Upsampling scale (2 == x2, 4 == x4, ...)')

    args = parser.parse_args()

    train = Train(args.input_dir, args.output_dir, args.model_dir, args.epochs, args.batch_size, args.scale)
    train.train()
