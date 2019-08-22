from keras.layers import concatenate
from keras.layers.convolutional import UpSampling2D, Convolution2DTranspose
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers import add


class Generator_Model(object):

    def __init__(self, noise_shape, out_shape):
        self.noise_shape = noise_shape
        self.out_shape = out_shape

    def generator(self):
        gen_model = []
        gen_input = Input(shape=[None, None, self.noise_shape[2]])
        bicubic_input = Input(shape=[None, None, self.out_shape[2]])
        filters_first_extraction = 196
        filters_extraction = 48
        filters_A1 = 64
        filters_B = 32
        model = Conv2D(filters=filters_first_extraction, kernel_size=3, strides=1, padding="same",
                       data_format="channels_last",
                       use_bias=True)(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                      shared_axes=[1, 2])(model)
        gen_model.append(model)

        for _ in range(7):
            model = Conv2D(filters=filters_extraction, kernel_size=3, strides=1, padding="same",
                           data_format="channels_last",
                           use_bias=True)(model)
            model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                          shared_axes=[1, 2])(model)
            gen_model.append(model)

        model = concatenate(gen_model)

        model_a1 = Conv2D(filters=filters_A1, kernel_size=1, strides=1, padding="same", data_format="channels_last",
                          use_bias=True)(model)
        model_a1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                         shared_axes=[1, 2])(model_a1)

        model_b1 = Conv2D(filters=filters_B, kernel_size=1, strides=1, padding="same", data_format="channels_last",
                          use_bias=True)(model)
        model_b1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                         shared_axes=[1, 2])(model_b1)

        model_b2 = Conv2D(filters=filters_B, kernel_size=3, strides=1, padding="same", data_format="channels_last",
                          use_bias=True)(model_b1)
        model_b2 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                         shared_axes=[1, 2])(model_b2)

        model = concatenate([model_a1, model_b2])

        model = Convolution2DTranspose(3, 3, strides=(2, 2), padding='same', data_format="channels_last",
                                       dilation_rate=(1, 1), activation='relu', use_bias=True,
                                       kernel_initializer='glorot_uniform', bias_initializer='zeros')(model)
        # model = Conv2D(filters=3, kernel_size=1, strides=1, padding="same", data_format="channels_last",
        #                use_bias=True)(final_con)
        # model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
        #               shared_axes=[1, 2])(model)
        # model = UpSampling2D(size=2)(model)
        model = add([model, bicubic_input])
        generator_model = Model(inputs=[gen_input, bicubic_input], outputs=model)

        return generator_model
