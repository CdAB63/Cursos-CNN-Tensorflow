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

BUFFER_SIZE = train_images.shape[0]
BATCH_SIZE  = 256

print('Buffer size: ', BUFFER_SIZE)
print('Batch  size: ', BATCH_SIZE)

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

###############################################################################
# MODELO
#
# Aqui criamos o modelo do gerador
#

from make_generator_model import make_generator_model

generator = make_generator_model()

# Imprima o modelo do gerador
generator.summary()

###############################################################################
# CRIE UMA IMAGEM PARA ADVERSARIAL
#

# CRIAMOS UM RUÍDO (noise)
noise = tf.random.normal([1, 100])

# APLICAMOS O RUÍDO AO GERADOR (ainda não treinado)
generated_image = generator(noise, training=False)

# OBTEMOS UMA IMAGEM GERADA
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.title('Imagem gerada com ruído aleatório')
plt.show()
