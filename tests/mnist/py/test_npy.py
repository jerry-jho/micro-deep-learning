import torch.nn.functional as f
import torch
import numpy as np
from img import load_image
import sys

np.set_printoptions(threshold=sys.maxsize)

im_data = load_image()
# print(im_data.numpy().transpose(0, 2, 3, 1))
conv0_weight = torch.from_numpy(np.load('conv0.0.weight.npy'))
# print(conv0_weight.numpy().transpose(0, 2, 3, 1))
conv0_bias = torch.from_numpy(np.load('conv0.0.bias.npy'))
# print(conv0_bias.numpy())
y = f.conv2d(im_data, conv0_weight, conv0_bias, stride=(1, 1), padding=(2, 2))
y = f.relu(y)

y = f.max_pool2d(y, kernel_size=2, stride=2, padding=0)

conv1_weight = torch.from_numpy(np.load('conv1.0.weight.npy'))
conv1_bias = torch.from_numpy(np.load('conv1.0.bias.npy'))

y = f.conv2d(y, conv1_weight, conv1_bias, stride=(1, 1), padding=(2, 2))
y = f.relu(y)

y = f.max_pool2d(y, kernel_size=2, stride=2, padding=0)
# print(y.numpy().transpose(0, 2, 3, 1))
y = y.view(y.size(0), -1)

out_weight = torch.from_numpy(np.load('out.weight.npy'))
out_bias = torch.from_numpy(np.load('out.bias.npy'))

y = f.linear(y, out_weight, out_bias)
print(y[0])
print(np.argmax(y))
