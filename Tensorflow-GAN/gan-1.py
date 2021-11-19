#!/bin/python
#-*- coding: utf-8 -*-

# IMPORTS
import tensorflow as tf
from tensorflow.keras import layers

###############################################################################
# LOAD DATASETS
# Aqui importamos o dataset mnist que consiste em imagens anotadas de dígitos
# 0-9 em caligrafia manual
###############################################################################

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
