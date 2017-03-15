'''
    Author: Roberto Matheus Pinheiro Pereira
    Minicurso: Aprendizagem Profunda aplicada a visão - JIM (2016)
    Developer at NCA - Nucleo de Computacao Aplicada 
    Date: 27/10/2016
    Available at: https://github.com/robertomatheuspp/JIM_2016_CNN/


  Description: 
  	- Use of Inception based CNN 
   	- Working with multiple branches
   	- Usage of merge and Merge
   	- Mutiple inputs and outputs
'''

import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, merge, Input
from keras.models import Sequential, Model
from keras.utils.visualize_util import plot
from keras.optimizers import SGD
from networks import cnn 


class SideNet(cnn.CNN):
  '''
    A linear layer with softmax loss as the classifier
    (predicting the same 1000 classes as the main classifier,
    but removed at inference time).
  '''
  def __init__(self, last_layer):
    # alternative output
    pool_alternative = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), border_mode="valid")(last_layer)
    conv_alternative = Convolution2D(128, 1, 1, border_mode="same")(pool_alternative)
    flat_alternative = Flatten()(conv_alternative)

    dense_alternative = Dense(1024, activation="relu")(flat_alternative)
    drop_alternative = Dropout(0.7)(dense_alternative)

    output_alternative = Dense(1000, activation="softmax")(drop_alternative)
    
    self.model = output_alternative
    


class Inception(cnn.CNN):
  ''' 
    GoogLeNet's Inception Model
  '''
  idx = 0

  def __init__(self, last_layer, nbf_conv_a=[64], nbf_conv_b=[96,128], nbf_conv_c=[16,32], nbf_pooling=[32]):
    Inception.idx += 1
    base_name = "incept({})".format(Inception.idx)
    conv_a = Convolution2D(nbf_conv_a[0], 1,1, activation='relu', border_mode="same",
                           name=base_name+"b_a0: 1x1({})+1(S)".format(nbf_conv_a[0])
                           )(last_layer)

    conv_b = Convolution2D(nbf_conv_b[0], 1,1, activation='relu', border_mode="same",
                           name=base_name + "b_a1: 1x1({})+1(S)".format(nbf_conv_b[0])
                           )(last_layer)
    conv_b = Convolution2D(nbf_conv_b[1] ,3,3, activation="relu", border_mode="same",
                           name=base_name + "b_a1: 3x3({})+1(S)".format(nbf_conv_b[1])
                           )(conv_b)

    conv_c = Convolution2D(nbf_conv_c[0], 1,1, activation='relu', border_mode="same",
                           name=base_name + "b_a2: 1x1({})+1(S)".format(nbf_conv_c[0])
                           )(last_layer)
    conv_c = Convolution2D(nbf_conv_c[1], 5,5,  activation="relu", border_mode="same",
                           name=base_name + "b_a2: 5x5({})+1(S)".format(nbf_conv_c[1])
                           )(conv_c)

    pooling = MaxPooling2D(pool_size=(3,3), strides=(1,1), border_mode="same",
                           name=base_name + "b_a3: 3x3+1(S)"
                           )(last_layer)

    pooling = Activation("relu",
                         name=base_name + ": relu"
                         )(pooling)

    pooling = Convolution2D(nbf_pooling[0], 1, 1, activation="relu", border_mode="same",
                            name=base_name + "b_a3: 1x1({})+1(S)".format(nbf_pooling[0])
                            )(pooling)

    self.model = merge([conv_a, conv_b, conv_c, pooling], mode="concat", concat_axis=1)#

    # return self.model


class Googlenet(cnn.CNN):
  def __init__(self, weights_path=None, heatmap=False):
    self.weights_path = weights_path
    inputs = Input( (3, 224, 224))
    # Convolution2D(nb_filter, nb_row, nb_col,  activation='linear', weights=None, border_mode='valid')
    conv1 = Convolution2D(64,
                             7,7,
                             activation='relu',
                             subsample=(2,2),
                            border_mode="same", name="7x7(64)+2(S)"
                          )(inputs)

    # MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default')
    pool1 = MaxPooling2D(pool_size=(3,3),
                             strides=(2,2),
                            border_mode="same", name="3,3+2(S)"
                             )(conv1)

    conv1_1 = Convolution2D(64,
                          1, 1,
                          activation='relu',
                          border_mode="valid", name="1x1(64)+1(V)"
                          )(pool1)

    conv2 = Convolution2D(192,
                            3,3,
                            activation='relu',
                            border_mode="same", name="3x3(192)+1(S)"
                            )(conv1_1)

    pool2 = MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),border_mode="same", name="p1_3x3+2(S)"
                           )(conv2)

    #inception 3a: [64], [96, 128], [16, 32], [32]
    inception_3a = Inception(pool2, nbf_conv_a=[64], nbf_conv_b=[96,128], nbf_conv_c=[16,32], nbf_pooling=[32]).getModel()
    #incetion 3b: [128], [128,192], [32,96], [64]
    inception_3b = Inception(inception_3a, [128], [128,192], [32,96], [64]).getModel()

    pool3 = MaxPooling2D(pool_size=(3,3),
                strides=(2,2),border_mode="same", name="p2_3x3+2(S)"
                         )(inception_3b)

    # inception (4a): [192],[96, 208], [16, 48], [64]
    inception_4a = Inception(pool3, [192],[96, 208], [16, 48], [64]).getModel()

    output1 = SideNet(inception_4a).getModel()


    # inception (4b): [160],[112, 224], [24, 64], [64]
    inception_4b = Inception(inception_4a, [160],[112, 224], [24, 64], [64]).getModel()
    # inception (4c): [128],[128, 256], [24, 64], [64]
    inception_4c = Inception(inception_4b, [128],[128, 256], [24, 64], [64]).getModel()
    # inception (4d): [112],[144, 288], [32, 64], [64]
    inception_4d = Inception(inception_4c, [112],[144, 288], [32, 64], [64]).getModel()
    # inception (4e): [256],[160, 320], [32, 128], [128]
    inception_4e = Inception(inception_4d, [256],[160, 320], [32, 128], [128]).getModel()

    output2 = SideNet(inception_4e).getModel()

    pool4 = MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2),border_mode="same")(inception_4e)
    inception_5a = Inception(pool4, [256], [160, 320], [32, 128], [128]).getModel()
    inception_5b = Inception(inception_5a, [384], [192, 384], [48, 128], [128]).getModel()

    pool5 = AveragePooling2D(pool_size=(7,7),border_mode="valid",
                                name="7x7+2(V)"
                             )(inception_5b)

    drop = Dropout(0.4)(pool5)
    flat = Flatten()(drop)
    dense = Dense(1000)(flat)
    activation = Activation("softmax")(dense)
    predictions = Dense(1000)(activation)


    self.model = Model(input=[inputs], output=[predictions, output1, output2])

    if weights_path and not heatmap:
    	self.model.load_weights(weights_path)
  
if __name__ == "__main__":
	googlenet = Googlenet().getModel()
	googlenet.compile(optimizer=SGD(momentum=0.9, decay=0.04),
	                  loss='categorical_crossentropy',
	                  metrics=['accuracy'])
	googlenet.summary()