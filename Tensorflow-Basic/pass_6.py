#!/usr/bin/python

###############################################################################
# Import libraries

# TENSORFLOW
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping

# AUXILIARY
import numpy as np
import pandas as pd
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

# Choose model
x = 0

while x != 1 and x != 2:
    print('Escolhla o modelo: ')
    print(' 1 -> last')
    print(' 2 -> best')
    print('Sua escolha: ',end='')
    try:
        x = int(input())
    except:
        print('Please, enter a value between 0 or 1 or 1')
        continue

# Load model
if x == 1:
    model = tf.keras.models.load_model('pass_5_last.h5')
else:
    model = tf.keras.models.load_model('pass_5_best.h5')

# Show model
model.summary()

print('Pressione qualquer botão para continuar: ', end='')
input()

# Loop para previsão do modelo
while True:
    # Predict values
    print('Forneça um número de amosta (-1 para encerrar): ', end='')
    n = int(input())
    if n == -1:
        quit()

    predictions = model.predict(np.expand_dims(X_test[n], axis=0))
    print(predictions.shape)
    i = 0
    for p in predictions[0]:
        print(cifar10_categories[i],':', p)
        i += 1
    idx = np.argmax(predictions[0])
    cat = cifar10_categories[idx]

    plt.imshow(X_test[n])
    plt.title(cat)
    plt.show()
