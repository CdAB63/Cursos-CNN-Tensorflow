#!/usr/bin/python

###############################################################################
# Import libraries

# TENSORFLOW
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

# AUXILIARY
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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
#model = tf.keras.Sequential()
model = tf.keras.Sequential()
# A convolutional layer 32x32x3 activation=ReLU, with padding
#model.add(tf.keras.layers.Conv2D(32, (3,3), padding='valid', activation='relu', input_shape=(32,32,3)))
model.add(tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', input_shape=(32,32,3)))
#model.add(tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', input_shape=(32,32,3)))
# Do a maxpool 2x2 stride=2
model.add(tf.keras.layers.MaxPooling2D((2,2), strides=2))
#model.add(tf.keras.layers.Conv2D(48, (3, 3), padding='valid', activation='relu'))
model.add(tf.keras.layers.Conv2D(192, (3, 3), padding='valid', activation='relu'))
#model.add(tf.keras.layers.Conv2D(384, (3, 3), padding='valid', activation='relu'))
# Do a maxpool 2x2 stride=2
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2))
# Now flatten to feed the ANN (fully connected/dense)
model.add(tf.keras.layers.Flatten())
# Now define the ANN (fully connected/dense)
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Add callbacks
# The idea of callbacks: have control over the training process
# Set to stop if Training does not improve after (10) epochs
callbacks = [EarlyStopping(patience=10)]

# Create a callback to sabe the best values
# This callback, an instance of ModelCheckpoint will save the state
# for the best val_accuracy so, every time the program gets a value
# for val_accuracy that's better than previous ones it will save it
# (mode='max') and discard last saved (save_best_only)
# It will save weights AND structure (save_weights_only=False)
model_checkpoint = ModelCheckpoint(filepath='pass_5_best.h5',
                                   save_weights_only=False,
                                   monitor='val_accuracy',
                                   mode='max',
                                   save_best_only=True)
callbacks.append(model_checkpoint)

# Compile the model
#
# During the training of the model, we tune the parameters(also known as hyperparameter tuning) 
# and weights to minimize the loss and try to make our prediction accuracy as correct as possible. 
# Now to change these parameters the optimizer’s role came in, which ties the model parameters 
# with the loss function by updating the model in response to the loss function output. 
# Simply optimizers shape the model into its most accurate form by playing with model weights. 
# The loss function just tells the optimizer when it’s moving in the right or wrong direction.
#      Adadelta: Optimizer that implements the Adadelta algorithm.
#      Adagrad: Optimizer that implements the Adagrad algorithm.
#  (*) Adam: Optimizer that implements the Adam algorithm.
#      Adamax: Optimizer that implements the Adamax algorithm.
#      Ftrl: Optimizer that implements the FTRL algorithm.
#      Nadam: Optimizer that implements the NAdam algorithm.
#      Optimizer: Base class for Keras optimizers.
#      RMSprop: Optimizer that implements the RMSprop algorithm.
#      SGD: Gradient descent (with momentum) optimizer.
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
#      
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_test,y_test),
                    callbacks=callbacks)

# Make training history intelligible
met_df1 = pd.DataFrame(history.history)
print(met_df1)
met_df1[['accuracy', 'val_accuracy']].plot()
plt.xlabel('Epocks')
plt.ylabel('Accuracy')
plt.title('Accuracies per Epoch')
plt.show()

# Save the model
model.save('pass_5_last.h5')
