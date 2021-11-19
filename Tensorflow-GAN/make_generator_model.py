#!/usr/bin/python
#-*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model():
    
    # Sempre começamos criando um MODELO
    # No caso será Sequential() (layer after layer)
    model = tf.keras.Sequential()

    # Então populamos o modelo adicionando LAYERS
    # No caso do GERADOR criamos uma rede convolucional invertida
    # onde as camadas de perceptrons estão na entrada
    # e as camadas convolucionais (que fornecem features) estão
    # na saída

    # Há uma série de operações que corrigem o shape dos dados
    # entre as camadas adicionadas

    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Notem que NESTE CASO PARTICLAR utilizamos na camada de saída uma função de ativação
    # que não é ReLU, mas é tanh, ilustrando mais uma vez a forma como podemos especificar
    # as funções de ativação.
    # 
    # Em aula futura posso explicar a escolha de tanh

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
