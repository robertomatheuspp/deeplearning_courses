'''
	Author: Roberto Matheus Pinheiro Pereira
	Minicurso: Aprendizagem Profunda aplicada a visão - JIM (2016)
	Developer at NCA - Nucleo de Computacao Aplicada 
	Date: 27/10/2016
	Available at: https://github.com/robertomatheuspp/JIM_2016_CNN/


	Description: 
		- Use of LeNet Convolutional Neural Network for MNIST database classification.
		- Introduction to keras 
			- Sequential
			- Convolution	
			- Pooling 
			- Model
				- Compile
				- Trainning
				- Testing 
				- Prediction x Evaluation
			- Model Visualisation
		- MNIST database
'''

import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras.utils.visualize_util import plot
from keras.datasets import mnist
from keras.utils import np_utils

from networks import cnn 


class Lenet(cnn.CNN):
	'''
		Implementation of LeNet (1998) convolutional neural network.
		It is an extension of CNN class.
	'''

	# Static and default Configuration 
	batch_size = 128
	nb_classes = 10
	nb_epoch = 100

	# input image dimensions
	img_rows, img_cols = 28, 28

	# size of pooling area for max pooling
	nb_pool = 2


	def __init__(self):
		#write lenet's model

		pass
		

def get_mnist_database():
	# treating input data
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train = X_train.reshape(X_train.shape[0], 1, Lenet.img_rows, Lenet.img_cols)
	X_train = X_train.astype('float32')
	X_train /= 255

	X_test = X_test.reshape(X_test.shape[0], 1, Lenet.img_rows, Lenet.img_cols)
	X_test = X_test.astype('float32')
	X_test /= 255

	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)

	return (X_train, y_train), (X_test, y_test)



if __name__ == "__main__":
	weights_path = "./weights/lenet_weights.h5"
	lenet = Lenet()
	#summary/plot model

	#compile

	#fit

	#load weigths

	#evaluate

	#predict



	