#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various utility functions for style transfer.
"""

import numpy as np

import keras.backend as K

def mean_subtraction(x):
    '''
    Subtracts the ImageNet mean value from given image.

    Parameters:
    -----------
        x : ndarray
            The image to subtract the mean from.

    Returns
    -------
        ndarray
        The mean subtracted image.
    '''
    imagenet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return x - imagenet_mean

def gram(x):
    '''
    Returns the Gram matrix of the given matrix. The Gram matrix is defined as:
        x * transpose(x)
    In order to calculate it, we first flatten the array in such a way so that
    its final shape is: (n_channels, x*y_dimensions).

    Parameters:
    -----------
        x : tensor
            The input array.

    Returns:
    --------
        tensor
        The Gram array of the given matrix.
    '''
    x_flat = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    x_num_elements = x.get_shape().num_elements()
    return K.dot(x_flat, K.transpose(x_flat)) / x_num_elements

def mean_addition(x, s):
    '''
    Undoes the mean subtraction in order to properly view the image.
    '''
    imagenet_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return np.clip(x.reshape(s)[:, :, :, ::-1] + imagenet_mean, 0, 255)

    
