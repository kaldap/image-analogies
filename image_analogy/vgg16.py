import os

import h5py
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    AveragePooling2D, Convolution2D, MaxPooling2D, ZeroPadding2D)
from tensorflow.keras.models import Sequential
from image_analogy import img_utils

def img_from_vgg(x):
    '''Decondition an image from the VGG16 model.'''
    x = x.transpose((1, 2, 0))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:,:,::-1]  # to RGB
    return x


def img_to_vgg(x):
    '''Condition an image for use with the VGG16 model.'''
    x = x.astype(np.float)
    x = x[:,:,::-1]  # to BGR
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    x = x.transpose((2, 0, 1))
    return x


def get_model(img_width, img_height, weights_path='vgg16_weights.h5', pool_mode='avg'):
    assert pool_mode in ('avg', 'max'), '`pool_mode` must be "avg" or "max"'
    if pool_mode == 'avg':
        pool_class = AveragePooling2D
    else:
        pool_class = MaxPooling2D

    pool_pad_mode = 'valid'
    conv_pad_mode = 'valid'

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_height, img_width)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding=conv_pad_mode, name='block1_conv1', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu', padding=conv_pad_mode, name='block1_conv2', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(pool_class((2, 2), strides=(2, 2), padding=pool_pad_mode))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding=conv_pad_mode, name='block2_conv1', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding=conv_pad_mode, name='block2_conv2', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(pool_class((2, 2), strides=(2, 2), padding=pool_pad_mode))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding=conv_pad_mode, name='block3_conv1', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding=conv_pad_mode, name='block3_conv2', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding=conv_pad_mode, name='block3_conv3', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(pool_class((2, 2), strides=(2, 2), padding=pool_pad_mode))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding=conv_pad_mode, name='block4_conv1', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding=conv_pad_mode, name='block4_conv2', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding=conv_pad_mode, name='block4_conv3', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(pool_class((2, 2), strides=(2, 2), padding=pool_pad_mode))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding=conv_pad_mode, name='block5_conv1', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding=conv_pad_mode, name='block5_conv2', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding=conv_pad_mode, name='block5_conv3', kernel_initializer="zeros", bias_initializer="zeros"))
    model.add(pool_class((2, 2), strides=(2, 2), padding=pool_pad_mode))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)

    # model.load_weights(weights_path)

    assert os.path.exists(weights_path), 'Model weights not found (see "--vgg-weights" parameter).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        layer = model.layers[k]
        if isinstance(layer, Convolution2D):
            weights[0] = np.array(weights[0])[:, :, ::-1, ::-1]
            weights[0] = img_utils.reshape_weights(weights[0])
        layer.set_weights(weights)

    #f.close()
    return model
