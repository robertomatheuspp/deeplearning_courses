# Author: Roberto Matheus Pinheiro Pereira
# Minicurso: Aprendizagem Profunda aplicada a visão - JIM (2016)
# Developer at NCA - Nucleo de Computacao Aplicada 
# Date: 27/10/2016
# Available at: https://github.com/robertomatheuspp/JIM_2016_CNN/
# Adapted from: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

import numpy as np
import time
# from keras.applications import vgg16, resnet50
from keras import backend as K
import sys
from math import sqrt, ceil
# from keras.utils.visualize_util import plot



# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def filtersFormLayer(layer, input_img,  filterOutSize, nb_filters=None, nb_filtersOut=None, nb_steps=20, input_depth=3):
    """
        based on: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
        
        Returns the inputs that maximize the activation of each filters in the given layer.


        Parameters:
            layer: the layer to be used. It can be acquired using model.layers[idx].
            input_img: is the placeholder for the input images. It is acquired using 'model.input'
            filterOutSize: size of image to be created for each filter.
            nb_filters: quantity of filters in the given layer to be analised. If None it will be created an image for every filter in the layer. 
            nb_filtersOut: quantity of filters in the given layer to be returned (based on the loss value). If None it will be returned nb_filters images. 
            nb_steps: number of epochs for the output images to be 'trainned' using gradient ascent.
    """

    if nb_filters == None or nb_filters > layer.output_shape[1]:
        nb_filters = layer.output_shape[1] - 1

    if nb_filtersOut == None or nb_filtersOut > layer.output_shape[1]:
        nb_filtersOut = nb_filters


    kept_filters = []
    for filter_index in range(0, nb_filters + 1):
        # we only scan through the first 'nb_filters' filters,
        # but there many be more of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer.output
        if K.image_dim_ordering() == 'th':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])
        # iterate = K.function([input_img, K.learning_phase()], [loss, grads])


        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        if K.image_dim_ordering() == 'th':
            input_img_data = np.random.random((1, input_depth, filterOutSize, filterOutSize))
        else:
            input_img_data = np.random.random((1, filterOutSize, filterOutSize, input_depth))
        input_img_data = (input_img_data - 0.5) * 20 + filterOutSize


        # we run gradient ascent for 'nb_steps'
        for i in range(nb_steps):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            sys.stdout.write('\t{}%: Current loss value: {}\r'.format(  100*(i+1)/nb_steps, loss_value))
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break
        # decode the resulting input image
        if loss_value > 1:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('\nFilter %d processed in %ds' % (filter_index, end_time - start_time))
    ## end for ##
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:nb_filtersOut]

    return kept_filters

def filtersToImage(filters, margin=5):
    '''
        Returns a grid image with all the filters in it.

        margin: distance of one filter image to another. (margin)
    '''
    filter_size = len(filters[0][0])
    nb_filters = len (filters) 
    nb_grid = int(ceil(sqrt(nb_filters)))

    # build a black picture with enough space for nb_grid and margin
    width = filter_size*nb_grid + margin*(nb_grid - 1)
    height = width
    img_out = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(nb_grid):
        for j in range(nb_grid):
            if (i*nb_grid + j >= nb_filters):
                break
            img, loss = filters[i * nb_grid + j]
            img_out[(filter_size + margin) * i: (filter_size + margin) * i + filter_size,
                     (filter_size + margin) * j: (filter_size + margin) * j + filter_size, :] = img

    return img_out



# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
# layer_name = 'block5_conv1'

# # build the VGG16 network with ImageNet weights
# model = vgg16.VGG16(weights='imagenet', include_top=True)
# # model = resnet50.ResNet50(weights='imagenet', include_top=True)
# print('Model loaded.')
# # Visualisation to file
# plot(model, to_file='./vgg-19.png')
# model.summary()

# # get the symbolic outputs of each "key" layer (we gave them unique names).
# layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
# filters_out = filtersFormLayer(layer=layer_dict[layer_name], input_img=model.input, filterOutSize=128, nb_filters=16, nb_steps=20)

# filtersToImage(filters_out)

