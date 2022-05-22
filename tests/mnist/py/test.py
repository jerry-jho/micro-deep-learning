import torch
from net import cnn
import numpy as np
from img import load_image

im_data = load_image()

Cnn = torch.load('cnn.pth')
Cnn.eval()
with torch.no_grad():
    testoutp, lstlayr = Cnn(im_data)
    print(testoutp[0])
    print(np.argmax(testoutp[0]))
