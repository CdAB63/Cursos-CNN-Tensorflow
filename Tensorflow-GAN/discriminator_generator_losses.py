#!/usr/bin/python
#-*- coding: utf-8 -*-

import tensorflow as tf

###############################################################################
# FUNÇÃO AUXILIAR PARA CALCULAR CROSS ENTROPY LOSS
#
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

###############################################################################
# PERDA NO DISCRIMINADOR (REAL/FAKE)
# A perda no discriminador é a soma das perdas (erros) no real e no fake
#
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

###############################################################################
# PERDA NO GERADOR
# A perda no gerador é simplesmente a perda no fake
#
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
