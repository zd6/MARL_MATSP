import numpy as np
def conv_shape_calc(raw_shape, kernal_size, stride, padding):
    H_in, W_in, ch = raw_shape
    if isinstance(kernal_size, int):
        kernal_size = (kernal_size, kernal_size)
    elif is_2d_shape(kernal_size):
        pass
    else:
        raise ValueError
    if isinstance(stride, int):
        stride = (stride, stride)
    elif is_2d_shape(stride):
        pass
    else:
        raise ValueError
    if isinstance(padding, int):
        padding = (padding, padding)
    elif is_2d_shape(padding):
        pass
    else:
        raise ValueError
    H_out = int(np.floor((H_in+2*padding[0]-1*(kernal_size[0]-1)-1)/stride[0]+1))
    W_out = int(np.floor((W_in+2*padding[1]-1*(kernal_size[1]-1)-1)/stride[1]+1))
    return (H_out, W_out,ch)

def is_2d_shape(kernal_size):
    return isinstance(kernal_size, list) and len(kernal_size)==2 and isinstance(kernal_size[0], int) and isinstance(kernal_size[1], int)