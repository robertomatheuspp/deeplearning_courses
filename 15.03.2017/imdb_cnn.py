# Reference: https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py

import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, Flatten, MaxPooling1D
from keras.layers import MaxPooling1D
from keras.datasets import imdb
from keras.utils.visualize_util import plot


#parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# Embedding layer
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

# Stack Convolution1D on the top of Embedding layer
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# Usage of a Pooling
# model.add(MaxPooling1D(3))
model.add(GlobalMaxPooling1D())
# model.add(Flatten())

# Artificial Neural Network:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))



# Compiling
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
plot(model, show_shapes=True)

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))



Y_prediction = model.predict_classes(X_test)
print ("prediction" , Y_prediction)
evaluation = model.evaluate (X_test, y_test, batch_size=batch_size)

print ("evaluation", evaluation)
print (model.metrics_names)