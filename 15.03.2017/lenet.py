# reference: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Activation


nb_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test  = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

#convolution
model.add(Convolution2D(nb_filters=32,
						nb_rows=3,
						nb_cols=3, 
						input_shape=input_shape))
#max pooling
model.add(MaxPooling2D(pool_size=(2,2)))
#convolution
model.add(Convolution2D(nb_filters=32,nb_rows=3,nb_cols=3))
#max pooling
model.add(MaxPooling2D(pool_size=(2,2)))

#flatten
model.add(Flatten())

#dense
model.add(Dense(50))
#dropout
model.add(Dropout(0.6))
#dense
model.add(Dense(nb_classes))
#activation
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
				              optimizer="adadelta",
				              metrics=['accuracy'])
			
model.fit(x_train, y_train, batch_size=128, nb_epoch=2)#, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])