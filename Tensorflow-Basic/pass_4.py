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

# Model creation Sequential -> layer after layer
model = tf.keras.Sequential()
# A convolutional layer 32x32x3 activation=ReLU, with padding
# --------- 32 filters ----------*
# --------- kernel/filter size --------*
# --------- input is a np object and not a tensor, so we supply the shape --------*
model.add(tf.keras.layers.Conv2D(32, (3,3), padding='valid', activation='relu', input_shape=(32,32,3)))
# Do a maxpool 2x2 stride=2
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))
# Another convolutional
model.add(tf.keras.layers.Conv2D(48, (3, 3), padding='valid', activation='relu'))
# Do a maxpool 2x2 stride=2
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2))
# Now flatten to feed the ANN (fully connected/dense)
model.add(tf.keras.layers.Flatten())
# Now define the ANN (fully connected/dense)
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Show model
model.summary()
