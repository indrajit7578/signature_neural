# -*- coding: utf-8 -*-
"""

@author: indra
"""

import numpy as np

np.random.seed(1337)  # for reproducibility
import os
import argparse

# from keras.utils.visualize_util import plot

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
import getpass as gp
import sys
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import random

random.seed(1337)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network_signet(input_shape):
    seq = Sequential()
    seq.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', subsample=(4, 4), input_shape=input_shape,
                          init='glorot_uniform', dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2), dim_ordering='tf'))

    seq.add(Convolution2D(256, 5, 5, activation='relu', name='conv2_1', subsample=(1, 1), init='glorot_uniform',
                          dim_ordering='tf'))
    seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))  # added extra
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

    seq.add(Convolution2D(384, 3, 3, activation='relu', name='conv3_1', subsample=(1, 1), init='glorot_uniform',
                          dim_ordering='tf'))
    seq.add(ZeroPadding2D((1, 1), dim_ordering='tf'))

    seq.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', subsample=(1, 1), init='glorot_uniform',
                          dim_ordering='tf'))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))  # added extra
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))
    seq.add(Dropout(0.5))

    seq.add(Dense(128, W_regularizer=l2(0.0005), activation='relu', init='glorot_uniform'))  # softmax changed to relu

    return seq





def get_test_image(height,width):
    file1 = 'f1.png'
    file2 = 'test2.png'
    tmp_img_dir = 'test/'
    image_pairs = []
    img1 = image.load_img(tmp_img_dir + file1, grayscale=True,
                          target_size=(height, width))

    img1 = image.img_to_array(img1)
    img1 = standardize(img1)

    img2 = image.load_img(tmp_img_dir + file2, grayscale=True,
                          target_size=(height, width))

    img2 = image.img_to_array(img2)
    img2 = standardize(img2)

    image_pairs += [[img1, img2]]
    images = [np.array(image_pairs)[:, 0], np.array(image_pairs)[:, 1]]

    return images

def standardize(img):
    img /= (20.201297 + 1e-7)
    return img

def main():

    img_height = 155
    img_width = 220

    input_shape = (img_height, img_width, 1)



    # network definition
    base_network = create_base_network_signet(input_shape)

    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    fname = os.path.join('model', 'weights1_' + 'CEDAR1' + '.hdf5')

    # load the best weights for test
    model.load_weights(fname)
    print(fname)
    print('Loading the best weights for testing done...')
    print("Predicting Start")
    test_res = model.predict(get_test_image(img_height, img_width), verbose=1)
    print("Prediction Result: ", test_res)




# Main Function
if __name__ == "__main__":
    main()
