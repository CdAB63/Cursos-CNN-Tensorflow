#!/usr/bin/python
#-*- coding: utf-8 -*-

import PIL

def display_image(image_index):
    return PIL.Image.open('./images/image_at_epoch_{:04d}.png'.format(image_index))
