import numpy as np

from keras.optimizers import SGD
from keras.applications import vgg16
from keras.utils.visualize_util import plot
from tools.convnetskeras.imagenet_tool import synset_to_dfs_ids
from networks.vgg_19 import VGG_19
from networks.lenet import Lenet

from scipy.misc import imread, imresize, imsave
from tools import filtermap
import matplotlib.pyplot as plt


def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We permute the colors to get them in the BGR order
        if color_mode=="bgr":
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        # We normalize the colors with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:,(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                      ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]

        img_list.append(img)

    img_batch = np.stack(img_list, axis=0)
    if not out is None:
        out.append(img_batch)
    else:
        return img_batch

if __name__ == "__main__":

    layer_name = "block5_conv1"
    #load cnn model or keras model


    #get model summary 

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    
    #produce filters
    

    #produce grid image from filters
    
    
    # save the result to disk
    