import numpy as np
from .interpolation import (
    nearest_neighbor_downsample, 
    bilinear_interpolation_downsample,
    bicubic_interpolation_downsample, 
    pixel_unshuffle_downsample    
)

from .conv2d import (
    generate_2dkernel, conv2d_downsample
)

from .pooling import fractional_pool2d

def downsample(img, scale_factor, ds_name=None):
    assert ds_name is not None 
    img = img.astype(np.float32)
    if ds_name.lower() in ['neighbor', 'bilinear', 'bicubic']:
        downsample_dict = {
            "neighbor": nearest_neighbor_downsample,
            "bilinear": bilinear_interpolation_downsample,
            "bicubic": bicubic_interpolation_downsample,
        }
        new_image = downsample_dict[ds_name](img, scale_factor)
        return new_image
    elif 'shuffle' in ds_name.lower():
        return pixel_unshuffle_downsample(img, scale_factor, return_list=False)
    elif 'pooling' in ds_name: # {mode}_pooling
        pooling_modes = ['max', 'min', 'avg', 'l2']
        mode = ds_name.split('_')[0].lower()
        assert mode in pooling_modes, f"Unimplemented pooling {mode}"
        return fractional_pool2d(img, scale_factor, mode=mode)
    elif 'conv2d' in ds_name: # {kernel}_pooling
        kernels_config = {
            'random': {'type': 'random', 'size': 5},
            'mean': {'type': 'mean', 'size': 5},
            'gaussion': {'type': 'gaussian', 'size': 5},
            'lanczos': {'type': 'lanczos', 'size': 5, 'a': 2}
        }
        kernel_name = ds_name.split('_')[0].lower()
        assert kernel_name in kernels_config, f"Unimplemented conv kernel {kernel_name}"
        kernel = generate_2dkernel(**kernels_config[kernel_name])
        return conv2d_downsample(img, scale_factor, kernel)