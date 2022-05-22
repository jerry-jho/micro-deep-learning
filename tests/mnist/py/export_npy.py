import torch
from net import cnn
import numpy as np
Cnn = torch.load('cnn.pth')
state_dict = Cnn.state_dict()
for k, v in state_dict.items():
    print(k)
    print(v.shape)
    np.save(k, v.numpy())