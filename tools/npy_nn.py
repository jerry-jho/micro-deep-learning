import torch.nn.functional as f
import torch
import numpy as np


def udl_conv2d_layer(desc: dict, ifm: np.ndarray, ker: np.ndarray,
                     vbs: np.ndarray) -> np.ndarray:
    # ifm source:     b,h,w,c
    # ifm torch:      b.c.h.w
    ifm_th = torch.from_numpy(ifm.transpose(0, 3, 1, 2))

    # ker source: co,kh,kw,ci
    # ker torch:  co,ci,kh,kw
    ker_th = torch.from_numpy(ker.transpose(0, 3, 1, 2))

    ofm_th = f.conv2d(ifm_th,
                      ker_th,
                      bias=None,
                      stride=(desc['sy'], desc['sx']),
                      padding=(desc['py'], desc['px']))

    # ofm torch: b, co, h, w
    # ofm source: b, h, w, co
    ofm_np = ofm_th.numpy().transpose(0, 2, 3, 1)
    
    if vbs is not None:
        ofm_np = ofm_np * vbs[0] + vbs[1]

    if desc['active'] == 'active_type_relu':
        ofm_np[ofm_np < 0] = 0

    return ofm_np


def udl_maxpooling2d_layer(desc: dict, ifm: np.ndarray) -> np.ndarray:
    ifm_th = torch.from_numpy(ifm.transpose(0, 3, 1, 2))
    ofm_th = f.max_pool2d(ifm_th,
                          kernel_size=(desc['ky'], desc['kx']),
                          stride=(desc['sy'], desc['sx']),
                          padding=(desc['py'], desc['px']))
    ofm_np = ofm_th.numpy().transpose(0, 2, 3, 1)
    if desc['active'] == 'active_type_relu':
        ofm_np[ofm_np < 0] = 0
    return ofm_np


def udl_dense_layer(desc: dict, ifm: np.ndarray, ker: np.ndarray,
                     vbs: np.ndarray) -> np.ndarray:

    if ifm.ndim == 4:
        # ifm source:     b,h,w,c
        # ifm torch:      b.c.h.w
        # ifm = ifm.transpose(0, 3, 1, 2)
        shape = ifm.shape
        ifm_s = ifm.reshape(ifm.shape[0],
                            ifm.shape[1] * ifm.shape[2] * ifm.shape[3])
        ifm_th = torch.from_numpy(ifm_s)
    else:
        ifm_th = torch.from_numpy(ifm)

    ker_shape = ker.shape
    ker_th = torch.from_numpy(ker.T)
    # print(ifm_th.numpy())
    # print(ifm_th.shape)
    # print(ker_th.shape)
    ofm_th = torch.matmul(ifm_th, ker_th) * vbs[0] + vbs[1]

    ofm_np = ofm_th.numpy()
    if desc['active'] == 'active_type_relu':
        ofm_np[ofm_np < 0] = 0
    return ofm_np