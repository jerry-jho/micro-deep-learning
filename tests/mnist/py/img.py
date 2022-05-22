import torch
from PIL import Image
import numpy as np

def load_image():
    im = Image.open('8.jpg')
    im_data = torch.from_numpy(np.asarray(im).astype(np.float32).transpose(2, 0, 1))
    im_data = 1.0 - im_data[0, :, :] / 255.0
    im_data = im_data.reshape(1, 1, 28, 28)
    return im_data