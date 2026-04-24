""" Utility functions — PyTorch version. """
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image))
              for i, path in zip(labels, paths)
              for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images


## Loss functions
def mse(pred, label):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    return torch.mean((pred - label) ** 2)


def xent(pred, label):
    return F.cross_entropy(pred, label)


## Network helpers (kept for compatibility, but sinusoid tasks use FC only)
def normalize(inp, activation, reuse=False, scope=''):
    """Identity normalization — just apply activation."""
    if activation is not None:
        return activation(inp)
    return inp


def conv_block(inp, cweight, bweight, reuse, scope, activation=F.relu, max_pool_pad='valid', residual=False):
    """Perform conv, nonlinearity, and optional max pool."""
    conv_output = F.conv2d(inp, cweight, bias=bweight, stride=2, padding=1)
    normed = normalize(conv_output, activation, reuse, scope)
    return normed
