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

from make_generator_model import make_generator_model

generator = make_generator_model()

###############################################################################
# CRIE UMA IMAGEM PARA ADVERSARIAL
#

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.title('Imagem gerada com ruído aleatório')
plt.show()

###############################################################################
# USE O DISCRIMINADOR PARA CLASSIFICAR IMAGENS GERADAS
# (VERDADEIRO/FALSO)
#

from make_discriminator_model import make_discriminator_model

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print('Discriminador criado: ', decision)

###############################################################################
# IMPORTE AS FUNÇÕES DE PERDA
# Como vamos fazer o treinamento (por back propagation) precisamos definir as
# funções de perda tanto para o discriminador quanto para o gerador
# afinal o sistema é treinado com as duas redes (veja figura na apresentação)
#
import discriminator_generator_losses
from discriminator_generator_losses import discriminator_loss
from discriminator_generator_losses import generator_loss

###############################################################################
# OTIMIZADORES
#
generator_optimizer     = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

###############################################################################
# SALVAR PONTOS DE VERIFICAÇÃO
#

checkpoint_dir = './training_checkpoints'

# O diretório existe?
if os.path.exists(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        print('O caminho ',checkpoint_dir,' existe, mas não é diretório.')
        quit()
else:
    try:
        os.mkdir(checkpoint_dir)
    except:
        print('Não foi possível criar o diretório ',checkpoint_dir,'.')
        quit()

# Como sei que existe o checkpoint_dir...
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

print('Checkpoint criado: ', checkpoint)
