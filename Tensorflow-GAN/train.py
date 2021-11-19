#/usr/bin/python
#-*- coding: utf-8 -*-

import tensorflow as tf
import discriminator_generator_losses
from discriminator_generator_losses import generator_loss, discriminator_loss
from IPython import display
import time
import matplotlib.pyplot as plt
import os

###############################################################################
# Passo de treinamento:
#      * Cria uma imagem ruído (noise)
#      * Usa o gerador para criar uma imagem a partir do ruído
#      * Mede a qualidade do output real
#      * Mede a qualidade do output fake
#      * Faz o cálculo dos gradientes
#           + gerador (gen_loss)
#           + discriminador (disc_loss)
@tf.function
def train_step(images, generator, generator_optimizer, discriminator, discriminator_optimizer, batch_size, noise_dim):
    
    noise = tf.random.normal([batch_size, noise_dim])
    
    """


With eager execution enabled, Tensorflow will calculate the values of tensors as they occur in your code. 
This means that it won't precompute a static graph for which inputs are fed in through placeholders. This means 
to back propagate errors, you have to keep track of the gradients of your computation and then apply these 
gradients to an optimiser.

This is very different from running without eager execution, where you would build a graph and then simply use 
sess.run to evaluate your loss and then pass this into an optimiser directly.

Fundamentally, because tensors are evaluated immediately, you don't have a graph to calculate gradients and so 
you need a gradient tape. It is not so much that it is just used for visualisation, but more that you cannot 
implement a gradient descent in eager mode without it.

Obviously, Tensorflow could just keep track of every gradient for every computation on every tf.Variable. However, 
that could be a huge performance bottleneck. They expose a gradient tape so that you can control what 
areas of your code need the gradient information. Note that in non-eager mode, this will be statically 
determined based on the computational branches that are descendants of your loss but in eager mode there is no 
static graph and so no way of knowing.

    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss  = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    
    gradients_of_generator     = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


###############################################################################
# TREINAMENTO
#
# Basicamente, para o número de ciclos (epochs) especificado e para o batch de
# imagens neste ciclo, realize um passo de treinamento (train_step)
#
# 
def train(dataset, epochs, generator, generator_optimizer, discriminator, discriminator_optimizer, batch_size, noise_dim, seed, checkpoint, checkpoint_prefix):

    for epoch in range(epochs):

        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, generator_optimizer, discriminator, discriminator_optimizer, batch_size, noise_dim)

        # Produce images
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Tempo para época {:5d} é {:.4f} segundos'.format(epoch + 1, time.time() - start - 1))

    # Época final
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

###############################################################################
# Gere imagens e salve
#
def generate_and_save_images(model, epoch, test_input):

    if not os.path.exists('./images'):
        try:
            os.mkdir('./images')
        except:
            print('Falha na criação do diretório ./images')
            quit()
    elif not os.path.isdir('./images'):
        print('./images existe mas não é diretório')
        quit()

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    try:
        plt.savefig('./images/image_at_epoch_{:04d}.png'.format(epoch))
    except Exception as e:
        print('Falhla ao salvar imagem image_at_epoch.{:04d}.png'.format(epoch))
        print('Motivo: ',str(e))
        quit()

    plt.show(block=False)
    plt.pause(1)
    plt.close()
