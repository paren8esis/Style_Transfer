#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for Neural Style Transfer:
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import pickle
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from keras.models import Model
from keras.losses import mean_squared_error
import keras.backend as K

from VGG16_Avg import VGG16_Avg

import utils

# Define the data folders
content_path = os.path.join(os.getcwd(), 'images')
style_path = os.path.join(os.getcwd(), 'images')

# Define the content image to be used
content_img_path = os.path.join(content_path, 'content_bird.jpg')

# Define the style image to be used
style_img_path = os.path.join(style_path, 'style_picasso.jpg')

# Define the results folder
results_path = os.path.join(os.getcwd(), 'images', 'results', os.path.basename(content_img_path).split('.')[0] + '_' + os.path.basename(style_img_path).split('.')[0])
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

# Open the content and style images
content_img = Image.open(content_img_path)

style_img = Image.open(style_img_path)

# Define the size to be used later on
image_shape = content_img.size

# Resize style to be equal to content size
style_img = style_img.resize(image_shape, Image.BILINEAR)

# Convert Images to ndarrays
content_img_arr = utils.mean_subtraction(np.array(content_img))
style_img_arr = utils.mean_subtraction(np.array(style_img))

# Define the results subfolder according to the chosen layers
content_layer_name = 'block4_conv2'
style_layers_names = ['block1_conv1',
                      'block2_conv1',
                      'block3_conv1',
                      'block4_conv1',
                      'block5_conv1']
results_path = os.path.join(results_path, 'content-' + content_layer_name + '-style' + ''.join(['-' + x for x in style_layers_names]))
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

# Get the content CNN activations
vgg_model = VGG16_Avg(include_top=False, input_shape=(image_shape[1], image_shape[0], 3))
outputs = {l.name: l.output for l in vgg_model.layers}

content_layer = outputs[content_layer_name]
content_model = Model(inputs=vgg_model.input, outputs=content_layer)

#content_predictions = content_model.predict(content_img_arr[np.newaxis, :, :, :])
#with open(os.path.join(results_path, 'content_predictions'), 'wb') as f:
#    pickle.dump(content_predictions, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(results_path, 'content_predictions'), 'rb') as f:
    content_predictions = pickle.load(f)
content_predictions_tensor = K.variable(content_predictions)

# Get the style CNN activations
style_layers = [outputs[x] for x in style_layers_names]
style_model = Model(inputs=vgg_model.input, outputs=style_layers)

#style_predictions = [x for x in style_model.predict(style_img_arr[np.newaxis, :, :, :])]
#with open(os.path.join(results_path, 'style_predictions'), 'wb') as f:
#    pickle.dump(style_predictions, f, pickle.HIGHEST_PROTOCOL)
with open(os.path.join(results_path, 'style_predictions'), 'rb') as f:
    style_predictions = pickle.load(f)
style_predictions_tensor = [K.variable(x) for x in style_predictions]

# Define the loss functions
L_content = K.mean(mean_squared_error(content_predictions_tensor, content_layer))

w_style = [0.05, 0.2, 0.2, 0.25, 0.3]
L_style = [K.mean(mean_squared_error(utils.gram(style_predictions_tensor[x][0]),
                              utils.gram(style_layers[x][0]))*w_style[x]) for x in range(len(style_predictions_tensor))]
L_style = sum(L_style)

alpha = 0.1
beta = 1
L_total = (alpha * L_content) + (beta * L_style)

# Define the loss gradients
L_gradients = K.gradients(L_total, vgg_model.input)

loss_grads_function = K.function([vgg_model.input], [L_content, L_style, L_total]+L_gradients)

# Define the results folder for the final images
results_images_path = os.path.join(results_path, ''.join(str(x) + '-' for x in w_style))[:-1]
results_images_path = os.path.join(results_images_path, 'a' + str(alpha) + '_b' + str(beta))
if not os.path.exists(results_images_path):
    os.makedirs(results_images_path, exist_ok=True)

class Evaluator(object):
    def __init__(self, f, shape):
        self.f = f
        self.shape = shape

    def loss(self, x):
        self.loss_cont, self.loss_st, self.loss_value, self.grad_values = self.f([x.reshape((1, image_shape[1], image_shape[0], 3))])
        return self.loss_value.astype(np.float64)

    def grads(self, x):
        return self.grad_values.flatten().astype(np.float64)

    def get_losses(self):
        return self.loss_cont, self.loss_st

evaluator_obj = Evaluator(loss_grads_function, image_shape)

# Generate random image
noise_image = np.random.uniform(-2.5, 2.5, (image_shape[1], image_shape[0], 3))

def minimize_loss(evaluator, n_iter, x):
    losses = []
    for i in range(n_iter):
        x, min_x, info = fmin_l_bfgs_b(evaluator_obj.loss,
                                       x.flatten(),
                                       fprime=evaluator_obj.grads,
                                       maxfun=20)
        print(i+1, " - current loss: ", min_x)
        losses.append(min_x)
        x = np.clip(x, -127, 127)
        imsave(os.path.join(results_images_path,
                            'final_res_at_iteration_' + str(i) + '.png'),
               utils.mean_addition(x.copy(), (1, image_shape[1], image_shape[0], 3))[0])

    return x, losses

new_image, losses = minimize_loss(evaluator_obj, 10, noise_image)

with open(os.path.join(results_images_path, 'loss.txt'), 'a') as f:
    for l in losses:
        f.write(str(l))
        f.write('\n')

losses = []
with open(os.path.join(results_images_path, 'loss.txt'), 'r') as f:
    for l in f:
        losses.append(float(l))

fig = plt.figure("Loss")
plt.plot(losses)
plt.show()


loss_c, loss_s, _, _ = evaluator_obj.loss(noise_image)
print("Content loss: ", loss_c)
print("Style loss: ", loss_s)

fig, ax = plt.subplots()
def animate(i):
    ax.imshow(Image.open(os.path.join(results_images_path, 'final_res_at_iteration_' + str(i) + '.png')))
anim = animation.FuncAnimation(fig, animate, frames=10, interval=200)
plt.show()
