import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt


def fractional_pool2d(image, scale_factor, mode='max'):
    """
    Fractional 池化（动态调整池化区域），池化的大小为
    :param image: 输入图像（H, W）
    :param mode: 'max', 'avg', 'l2'
    :return: 池化后的图像
    """
    assert scale_factor > 1 , "scale factor must be greater than 1"
    # width, height, channels
    h, w  = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    new_w, new_h = int(w/scale_factor), int(h/scale_factor)
    # print(f"{mode} pooling: {h}x{w} -> {new_h}x{new_w}")

    if c == 1:
        new_image = np.zeros((new_h, new_w), dtype=image.dtype)
    else:
        new_image = np.zeros((new_h, new_w, c), dtype=image.dtype)
    
    for y in range(new_h):
        for x in range(new_w):
            # 计算动态池化区域
            h_start = max(0, int(np.floor(y * scale_factor)))
            h_end = min(int(np.ceil((y + 1) * scale_factor)), h)
            w_start = max(0, int(np.floor(x * scale_factor)))
            w_end = min(int(np.ceil((x + 1) * scale_factor)), w)
            
            patch = image[h_start:h_end, w_start:w_end]
            if mode.lower() == 'max':
                new_image[y, x] = np.max(patch, axis=(0,1))
            elif mode.lower() == 'min':
                new_image[y, x] = np.min(patch, axis=(0,1))
            elif mode.lower() == 'avg':
                new_image[y, x] = np.mean(patch, axis=(0,1))
            elif mode.lower() == 'l2':
                new_image[y, x] = np.sqrt(np.mean(patch ** 2, axis=(0,1)))
    return new_image