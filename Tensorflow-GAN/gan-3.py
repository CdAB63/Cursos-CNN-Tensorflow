#!/bin/python
#-*- coding: utf-8 -*-

# IMPORTS
import tensorflow as tf
from tensorflow.keras import layers

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from IPython import display

###############################################################################
# LOAD DATASETS
###############################################################################

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

###############################################################################
# NORMALIZATION
#
# Aqui fazemos algo diferente do exemplo anterior, onde normalizamos entre 0 e
# 1... Neste caso vamos normalizar as imagens entre -1.0 e 1.0
#
###############################################################################

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

###############################################################################
# LABELS
#
# Estamos trabalhando com números... então vai de '0' a '9'
#
###############################################################################

labels = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]

###############################################################################
# PREPARE O TRAINING SET
#
# BATCHES
# 
# Aqui dividimos o training set em "slices" e para evitar vies fazemos um 
# embaralhamento (shuffle)
#
# É algo equivalente ao que fazemos no darknet quando definimos os parâmetros
# 'batches' e 'subdivisions'. O darknet faz o shuffle (embaralhamento das
# imagens de treino automaticamente, mas ainda assim eu já forneço a lista de
# arquivos do training set embaralhada para aumentar a entropia
#

BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE  = 256

print('Buffer size: ', BUFFER_SIZE)
print('Batch  size: ', BATCH_SIZE)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print('Train dataset shape: ', train_dataset)

while True:
    print('Entre o número de um slice (imagem) -1 para terminar: ', end='')
    try:
        idx = int(input())
    except:
        print('Precisa ser número')
        continue
    if idx == -1:
        break
    try:
        img = list(train_dataset)[0][idx]
        plt.imshow(img)
        plt.title('Imagem #'+str(idx))
        plt.show()
    except Exception as e:
        print('Algo errado aconteceu. Índice fora do range?')
        print(str(e))
        continue
