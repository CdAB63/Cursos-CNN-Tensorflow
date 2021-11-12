#!/usr/bin/python

###############################################################################
# Import libraries

# TENSORFLOW
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# AUXILIARY
import matplotlib.pyplot as plt

###############################################################################
# CODE

# Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Hand write the names of categories
cifar10_categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]

# It's good to have everything between 0 and 1...
# otherwise it's from 0 to 255...
X_train = X_train/255
X_test  = X_test/255

print('X_train shape is: ',X_train.shape)
print('X_test  shape is: ',X_test.shape)
