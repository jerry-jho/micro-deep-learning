import torch.nn.functional as f
import torch
import numpy as np
import npy_nn
import sys
np.set_printoptions(threshold=sys.maxsize)

im_data = np.load('im_data.udl.npy')

conv0_weight = np.load('conv0.0.weight.udl.npy')
conv0_bias = np.load('conv0.0.bias.udl.npy')

y = npy_nn.udl_conv2d_layer(
    {
        'py': 2,
        'px': 2,
        'sy': 1,
        'sx': 1,
        'active': 'active_type_relu'
    }, im_data, conv0_weight, conv0_bias)

y = npy_nn.udl_maxpooling2d_layer(
    {
        'py': 0,
        'px': 0,
        'sy': 2,
        'sx': 2,
        'ky': 2,
        'kx': 2,
        'active': 'active_type_linear'
    }, y)

conv1_weight = np.load('conv1.0.weight.udl.npy')
conv1_bias = np.load('conv1.0.bias.udl.npy')

y = npy_nn.udl_conv2d_layer(
    {
        'py': 2,
        'px': 2,
        'sy': 1,
        'sx': 1,
        'active': 'active_type_relu'
    }, y, conv1_weight, conv1_bias)

y = npy_nn.udl_maxpooling2d_layer(
    {
        'py': 0,
        'px': 0,
        'sy': 2,
        'sx': 2,
        'ky': 2,
        'kx': 2,
        'active': 'active_type_linear'
    }, y)

out_weight = np.load('out.weight.udl.npy')
out_bias = np.load('out.bias.udl.npy')

y = npy_nn.udl_dense_layer(
    {
        'active': 'active_type_linear'
    }, y, out_weight, out_bias)
print(y[0])
print(np.argmax(y))