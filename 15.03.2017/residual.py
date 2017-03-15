from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,merge,BatchNormalization
from keras.models import Model
from keras.utils.visualize_util import plot


def identity(input_tensor):
	conv_name_base = 'res_branch_identity'
	bn_name_base = 'bn_branch_identity'
	bn_axis = 1
	x = Convolution2D(nb_filter=64, 
					  nb_col=1,
					  nb_row=1, 
					  name=conv_name_base + '2a')(input_tensor)
	# x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	# x = Activation('relu')(x)

	x = Convolution2D(nb_filter=64, 
					 nb_col=3,
					 nb_row=3,
	                 border_mode='same', 
	                 name=conv_name_base + '2b')(x)
	# x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	# x = Activation('relu')(x)

	x = Convolution2D(nb_filter=256, 
					  nb_row=1,
					  nb_col=1, 
					  name=conv_name_base + '2c')(x)
	# x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = merge([x, input_tensor], mode='sum')
	x = Activation('relu')(x)
	return x
def projection(prev_layer):
	conv_name_base = 'res_branch_proj'
	bn_name_base = 'bn_branch_proj'
	bn_axis = 1
	x = Convolution2D(nb_filter=64,  
					  nb_row=1,
					  nb_col=1, 
					  subsample=(2,2),
	                  name=conv_name_base + '2a')(prev_layer)
	# x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	# x = Activation('relu')(x)

	x = Convolution2D(nb_filter=32,  
					  nb_row=3,
					  nb_col=3, 
					  border_mode='same',
	                  name=conv_name_base + '2b')(x)
	# x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	# x = Activation('relu')(x)

	x = Convolution2D(nb_filter=256,  
					 nb_row=1,
					 nb_col=1, 
					 name=conv_name_base + '2c')(x)
	# x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Convolution2D(256,  1,1, subsample=(2,2),
	                         name=conv_name_base + '1')(prev_layer)
	# shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = merge([x, shortcut], mode='sum')
	x = Activation('relu')(x)
	return x

inputs=Input(shape=(1,256,256))
x = Convolution2D(nb_filter=256, nb_col=2,nb_row=2)(inputs)
x = identity(x)
x = projection(x)

model = Model(input=[inputs], output=[x])

plot (model, show_shapes=True)