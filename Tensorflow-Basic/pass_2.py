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

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

cifar10_categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]

X_train = X_train/255
X_test  = X_test/255

image = X_train[3]
plt.imshow(image)
plt.title(cifar10_categories[y_train[3][0]])
plt.show()
