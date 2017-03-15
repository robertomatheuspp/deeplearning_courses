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
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras.utils.visualize_util import plot
from keras.datasets import mnist
from keras.utils import np_utils
from tools import filtermap
from tools.convnetskeras.customlayers import Softmax4D

from networks import cnn 


#### LeNet CNN Model
class Lenet(cnn.CNN):
	# Default Configuration 
	batch_size = 128
	nb_classes = 10
	nb_epoch = 100

	# input image dimensions
	img_rows, img_cols = 28, 28

	# size of pooling area for max pooling
	nb_pool = 2


	def __init__(self, weights_path=None, heatmap=False):
		self.weights_path = weights_path
		self.canHeatMap = True
		model = Sequential()
		if heatmap: 
			model.add(Convolution2D(64, 5, 5,
			                      border_mode='valid',
			                      input_shape=(1, None, None), name="conv1"))
		else:
			model.add(Convolution2D(64, 5, 5,
			                      border_mode='valid',
			                      input_shape=(1, Lenet.img_rows, Lenet.img_cols), name="conv1"))
		model.add(MaxPooling2D(pool_size=(Lenet.nb_pool, Lenet.nb_pool)))
		model.add(Convolution2D(32, 3, 3, name="conv2"))
		model.add(MaxPooling2D(pool_size=(Lenet.nb_pool, Lenet.nb_pool)))

		if heatmap:
			model.add(Convolution2D(128, 5, 5, activation="relu", name="dense_1"))
			model.add(Convolution2D(10, 1, 1, activation="relu", name="dense_2"))
			model.add(Softmax4D(axis=1, name="softmax"))
		else:
			model.add(Dropout(0.25))
			model.add(Flatten())
			model.add(Dense(128))
			model.add(Activation('relu'))
			model.add(Dropout(0.5))
			model.add(Dense(Lenet.nb_classes))
			model.add(Activation('softmax'))

		if weights_path and not heatmap:
			model.load_weights(weights_path)
		self.model = model
			

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
	(X_train, y_train), (X_test, y_test) = get_mnist_database()

	# Instantiating CNN
	lenet = Lenet()
	model = lenet.getModel()

	# Model Compiling
	model.compile(loss='categorical_crossentropy',
	              optimizer="adadelta",
	              metrics=['accuracy'])

	# Visualisation to file
	# plot(model, to_file='./lenet.png')

	# Model Fitting
	# model.fit(X_train, y_train, batch_size=Lenet.batch_size, nb_epoch=Lenet.nb_epoch,
	#           verbose=1, validation_split=0.3)

	model.load_weights("./weights/lenet_weights.h5")
	# model.save_weights("./weights/lenet_weights.out")
	model.summary()

	# Evaluate
	evaluation = model.evaluate(X_test, y_test)
	print (evaluation)

	# Predicting
	# prediction = model.predict(X_test, verbose=0)
	
   # produceHeatMap()


