#!/usr/bin/python
#-*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

cifar10_categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' ]

print('Os labels (numéricos do training set)')
print(y_train)

print('A seguir os labels por nome do training set')
print('Pressione <enter> para continuar', end='')
input()

for cat in y_train:
    print('categoria: ',cifar10_categories[cat[0]])
