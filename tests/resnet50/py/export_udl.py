import h5py
import json
import numpy as np


def f32(f):
    s = str(f)
    s = s.split('.')[0] + '.' + s.split('.')[1][0:7]
    return s


def c_array(name, array):
    array = array.astype(np.float32).reshape(array.size)
    s = 'float %s[] = {' % name.replace('.', '_')
    sa = [str(x) for x in array]
    s += ','.join(sa) + '};'
    return s


layers = json.load(open('model.json'))['config']['layers']

wh5 = h5py.File('resnet50.h5', 'r')
fh5 = h5py.File('feature.h5', 'r')

h_arr = []
c_arr = []

for i, layer in enumerate(layers):
    layer_name = layer['name']
    print(layer_name)
    layer_cls = layer['class_name']
    hdf_key = 'model_weights/%s/%s' % (layer_name, layer_name)
    if layer_cls == 'Conv2D':
        k = np.array(wh5[hdf_key + '/kernel:0']).transpose(3, 0, 1, 2)
        b = np.array(wh5[hdf_key + '/bias:0'])
        L = b.shape[0]
        vb = np.ones((2, L), dtype=np.float32)
        vb[1, :] = b
        c_arr.append(c_array(layer_name + '_k', k))
        c_arr.append(c_array(layer_name + '_b', vb))
        h_arr.append("extern float %s_k[];" % layer_name)
        h_arr.append("extern float %s_b[];" % layer_name)
    elif layer_cls == 'Dense':
        k = np.array(wh5[hdf_key + '/kernel:0']).transpose(1, 0)
        b = np.array(wh5[hdf_key + '/bias:0'])
        L = b.shape[0]
        vb = np.ones((2, L), dtype=np.float32)
        vb[1, :] = b
        c_arr.append(c_array(layer_name + '_k', k))
        c_arr.append(c_array(layer_name + '_b', vb))
        h_arr.append("extern float %s_k[];" % layer_name)
        h_arr.append("extern float %s_b[];" % layer_name)
    elif layer_cls == 'BatchNormalization':
        g = np.array(wh5[hdf_key + '/gamma:0'])
        m = np.array(wh5[hdf_key + '/moving_mean:0'])
        v = np.array(wh5[hdf_key + '/moving_variance:0'])
        b = np.array(wh5[hdf_key + '/beta:0'])
        sp = 1.0 / np.sqrt(v + float(layer['config']['epsilon']))
        L = b.shape[0]
        vb = np.ones((2, L), dtype=np.float32)
        vb[0, :] = g * sp
        vb[1, :] = b - g * sp * m
        c_arr.append(c_array(layer_name + '_b', vb))
        h_arr.append("extern float %s_b[];" % layer_name)
    # if i > 40:
    #     break

c_arr.append(c_array('input', np.array(fh5['input_1'])))
h_arr.append("extern float input[];")
with open('data.c', 'w') as fp:
    fp.write('\n'.join(c_arr))
with open('data.h', 'w') as fp:
    fp.write('\n'.join(h_arr))