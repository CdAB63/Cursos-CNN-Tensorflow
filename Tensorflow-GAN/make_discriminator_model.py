#!/usr/bin/python
#-*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers

def make_discriminator_model():

    # Novamente criamos um modelo
    model = tf.keras.Sequential()

    # Este modelo é tradicional para CNNs ou seja, uma série de feature extractors
    # implementados por camadas convolucionais e terminando por uma rede
    # de perceptrons completamente conectada, no nosso caso 1 camada com 1 valor
    # de saída (probabilidade de pertencer à uma classe

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
