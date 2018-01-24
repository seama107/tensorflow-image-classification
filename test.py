import numpy as np
import tensorflow as tf
import dataset
import cv2
import os
from optparse import OptionParser

from cnn import *




parser = OptionParser()
parser.add_option("-i", "--input", dest="input_dir")
parser.add_option("-m", "--model", dest="model_to_load", default="models/model.ckpt")

(options, args) = parser.parse_args()
model_to_load = options.model_to_load
input_dir = options.input_dir


# Hyperparams

# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 256

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['Pool', 'NonPool']
num_classes = len(classes)

test_data = dataset.DataSet(*dataset.load_train(input_dir, img_size, classes))
print(test_data.images)
