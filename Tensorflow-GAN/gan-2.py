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
# É importante ter em mente que na maior parte dos casos as imagens de treino
# tem que ser condicionadas porque ou foram geradas para casos gerais ou
# foram geradas tendo em vista requisitos de outros sistemas
#
# No caso do mnist as imagens são B&W e 1 plano de cor
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

print('Training set (images) shape: ', train_images.shape)
print('Annotations shape: ', train_labels.shape)

###############################################################################
# MOSTRE IMAGENS DO DATASET
#
###############################################################################

while True:
    
    try:
        print('Entre um número de imagem (-1 para sair): ', end='')
        n = int(input())
    except:
        print('Por favor, informe um número válido...')
        continue

    if n == -1:
        quit()
    try:
        img = train_images[n]
        plt.imshow(img)
        plt.title(labels[train_labels[n]])
        plt.show()
    except:
        print('Imagem fora do intervalo... Escolha outra (com índice menor)')
