import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling1D, Activation, Embedding

from keras.datasets import imdb
from keras.preprocessing import sequence 

#load data
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=5000)

#pad sequence
X_train = sequence.pad_sequences(X_train, maxlen=400)
X_test = sequence.pad_sequences(X_test, maxlen=400)


#Building Model
model = Sequential()
model.add(Embedding(5000,
                    50,
                    input_length=400,
                    dropout=0.2))
#convolution
model.add(Convolution1D(nb_filter=32,
						filter_length=3))
#max pooling
model.add(MaxPooling1D(3))
#convolution
model.add(Convolution1D(nb_filter=32,
						filter_length=3))
#max pooling
model.add(MaxPooling1D(2))
#flatten
model.add(Flatten())

#dense
model.add(Dense(50))
#dropout
model.add(Dropout(0.6))
#dense
model.add(Dense(1))
#activation
model.add(Activation('softmax'))


#Compiling
model.compile(loss='binary_crossentropy',
	              optimizer="adam",
	              metrics=['accuracy'])
			
model.fit(X_train, y_train, batch_size=128, nb_epoch=2)#, validation_data=(x_test, y_test))
score = model.evaluate(X_test, y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])