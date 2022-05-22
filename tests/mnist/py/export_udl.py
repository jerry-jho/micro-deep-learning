import torch
from net import cnn
import numpy as np
from img import load_image


def save_array(filename, array):
    array = array.astype(np.float32)
    b = array.tobytes()
    with open(filename, 'wb') as f:
        f.write(b)


Cnn = torch.load('cnn.pth')
state_dict = Cnn.state_dict()
for k, v in state_dict.items():
    print(k)
    print(v.shape)
    np_array = v.numpy().astype(np.float32)
    if 'weight' in k:
        if np_array.ndim == 4:
            np_array = np_array.transpose(0, 2, 3, 1)
        else:
            np_array = np_array.reshape(10, 32, 7, 7)
            # co ci h w
            # co h w ci
            np_array = np_array.transpose(0, 2, 3, 1)
            np_array = np_array.reshape(10, 32 * 7 * 7)
    elif 'bias' in k:
        L = np_array.shape[0]
        new_array = np.ones((2, L), dtype=np.float32)
        new_array[1, :] = np_array
        # print(np_array)
        # print(new_array)
        np_array = new_array
    save_array(k + '.bin', np_array)
    np.save(k + '.udl', np_array)

im_data = load_image()
im_data = im_data.numpy().transpose(0, 2, 3, 1)
# print(im_data)
save_array('im_data.bin', im_data)
np.save('im_data.udl', im_data)
