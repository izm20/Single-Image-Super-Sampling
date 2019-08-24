from keras.layers import concatenate, BatchNormalization, Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D, Convolution2DTranspose
from keras.models import Model


class SRDeepCNN(object):

    def __init__(self, channels=3, scale=2, filters_first_extraction=196, filters_extraction=48, extraction_layers=11,
                 filters_a1=64, filters_b=32):
        self.channels = channels
        self.upsampling_n = int(scale // 2)
        self.filters_first_extraction = filters_first_extraction
        self.filters_extraction = filters_extraction
        self.extraction_layers = extraction_layers
        self.filters_a1 = filters_a1
        self.filters_b = filters_b

    def extraction_phase(self, model):
        model = Conv2D(filters=self.filters_extraction, kernel_size=3, strides=1, padding="same",
                       data_format="channels_last",
                       use_bias=True)(model)
        model = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                   beta_initializer='zeros', gamma_initializer='ones',
                                   moving_mean_initializer='zeros',
                                   moving_variance_initializer='ones')(model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                      shared_axes=[1, 2])(model)
        return model

    def reconstruction_a(self, model):
        model_a1 = Conv2D(filters=self.filters_a1, kernel_size=1, strides=1, padding="same",
                          data_format="channels_last",
                          use_bias=True)(model)
        model_a1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                         shared_axes=[1, 2])(model_a1)
        return model_a1

    def reconstruction_b(self, model):
        model_b1 = Conv2D(filters=self.filters_b, kernel_size=1, strides=1, padding="same", data_format="channels_last",
                          use_bias=True)(model)
        model_b1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                         shared_axes=[1, 2])(model_b1)
        model_b2 = Conv2D(filters=self.filters_b, kernel_size=3, strides=1, padding="same", data_format="channels_last",
                          use_bias=True)(model_b1)
        model_b2 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                         shared_axes=[1, 2])(model_b2)
        return model_b2

    def upsampling(self, model):
        model = Convolution2DTranspose(filters=3, kernel_size=3, strides=(2, 2), padding='same',
                                       data_format="channels_last", dilation_rate=(1, 1), activation='relu',
                                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(
            model)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                      shared_axes=[1, 2])(model)
        return model

    def build_model(self):
        gen_input = Input(shape=[None, None, self.channels])
        gen_model = []
        model = Conv2D(filters=self.filters_first_extraction, kernel_size=3, strides=1, padding="same",
                       data_format="channels_last",
                       use_bias=True)(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                      shared_axes=[1, 2])(model)
        gen_model.append(model)
        for _ in range(self.extraction_layers):
            model = self.extraction_phase(model)
            gen_model.append(model)
        model = concatenate(gen_model)
        model_a1 = self.reconstruction_a(model)
        model_b2 = self.reconstruction_b(model)
        model = concatenate([model_a1, model_b2])
        for _ in range(self.upsampling_n):
            model = self.upsampling(model)
        model = Model(inputs=[gen_input], outputs=model)
        return model
