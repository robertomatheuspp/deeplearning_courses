import numpy as np

from keras.layers import Input, Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import SGD
from scipy.misc import imread, imresize

from networks import cnn

from tools.convnetskeras.imagenet_tool import synset_to_dfs_ids
from tools.convnetskeras.customlayers import Softmax4D

import matplotlib.pyplot as plt

class VGG_19(cnn.HeatConvNet):
    def __init__(self, weights_path=None, heatmap=False):
        self.weights_path = weights_path
        self.canHeatMap = True
        model = Sequential()

        if heatmap:
            model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
        else:
            model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
        
        #adding the architecture's core
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_4'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_4'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_4'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        if heatmap:
            model.add(Convolution2D(4096,7,7,activation="relu",name="dense_1"))
            model.add(Convolution2D(4096,1,1,activation="relu",name="dense_2"))
            model.add(Convolution2D(1000,1,1,name="dense_3"))
            model.add(Softmax4D(axis=1,name="softmax"))
        else:
            model.add(Flatten())
            model.add(Dense(4096, activation='relu', name='dense_1'))
            model.add(Dropout(0.5))
            model.add(Dense(4096, activation='relu', name='dense_2'))
            model.add(Dropout(0.5))
            model.add(Dense(1000, name='dense_3'))
            model.add(Activation("softmax"))

        if weights_path and not heatmap:
            model.load_weights(weights_path)

        self.model = model


    
