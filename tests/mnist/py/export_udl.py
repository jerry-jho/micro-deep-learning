import torch
from net import cnn
import numpy as np
from img import load_image


def save_array(filename, array):
    array = array.astype(np.float32).reshape(array.size)
    s = 'float %s[] = {' % filename.replace('.', '_')
    sa = [str(x) for x in array]
    s += ','.join(sa) + '};'
    return s


cs = ''
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
    cs += save_array(k, np_array) + "\n"
    np.save(k + '.udl', np_array)

im_data = load_image()
im_data = im_data.numpy().transpose(0, 2, 3, 1)
# print(im_data)
cs += save_array('im_data.bin', im_data) + "\n"
np.save('im_data.udl', im_data)

with open('data.c', 'w') as fp:
    fp.write(cs)