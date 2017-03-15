from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,merge
from keras.models import Model
from keras.utils.visualize_util import plot

input_layer = Input(shape=(1,64,64), name='entrada')
#branch 1
conv_a = Convolution2D(nb_filter=64, 
					   nb_col=1,
					   nb_row=1, 
					   activation='relu', 
					   border_mode="same",
                       )(input_layer)
#branch 2
conv_b = Convolution2D(nb_filter=96,
					   nb_col=1,
					   nb_row=1, 
					   activation='relu', 
					   border_mode="same",
                       )(input_layer)
conv_b = Convolution2D(nb_filter=128,
					   nb_col=3,
					   nb_row=3,
					   activation="relu", 
					   border_mode="same",
                       )(conv_b)
#branch 3
conv_c = Convolution2D(nb_filter=16, 
						nb_col=1,
						nb_row=1, 
						activation='relu', 
						border_mode="same",
                       )(input_layer)
conv_c = Convolution2D(nb_filter=32, 
						nb_col=5,
						nb_row=5,  
						activation="relu", 
						border_mode="same",
                       )(conv_c)
#branch pooling
pooling = MaxPooling2D(pool_size=(3,3), 
						strides=(1,1), 
						border_mode="same"
                       )(input_layer)

pooling = Activation("relu",
                     )(pooling)

pooling = Convolution2D(nb_filter=32, 
						nb_col=1, 
						nb_row=1, 
						activation="relu", 
						border_mode="same",
                        )(pooling)

#concatenate
concatenation = merge([conv_a, conv_b, conv_c, pooling], mode="concat", concat_axis=1)

#creating model
model = Model(input=[input_layer], output=[concatenation])

#ploting
plot (model, show_shapes=True)