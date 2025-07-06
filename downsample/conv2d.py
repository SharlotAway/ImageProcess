import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt


def gaussion_kernel(size, sigma=None):
    if sigma is None or sigma < 0:
        sigma = 0.3*((size-1)*0.5-1)+0.8
    size = size + 1 if size % 2 == 0 else size 
    center = size // 2 

    assert size <= 7 and size >= 1 and size % 2 == 1, "size must be odd and in range [1, 7]"
    
    kernel_1d = np.zeros((size, 1), dtype=np.float64)
    sum_val = 0.0 
    scale2X = - 0.5 / (sigma ** 2)

    for i in range(size):
        x = i - center 
        kernel_1d[i,0] = np.exp(scale2X * (x ** 2))
        sum_val += kernel_1d[i, 0]
    
    inv_sum = 1.0 / sum_val 
    kernel_1d *= inv_sum

    kernel_2d = kernel_1d @ kernel_1d.T
    return kernel_2d 

# import cv2
# def cv2_gaussion_kernel(size, sigma=None):
#     sigma = 0.3*((size-1)*0.5-1)+0.8
#     return cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T


def generate_2dkernel(**kernel_kwargs):
    kernel_type = kernel_kwargs.get('type', 'gaussian').lower()
    size = kernel_kwargs.get('size', 3)

    if kernel_type == 'random':
        kernel = np.random.rand(size, size)
    elif kernel_type == 'mean':
        kernel = np.ones((size, size)) / (size ** 2)
    elif kernel_type == 'gaussian':
        assert size <= 7, "Gaussion kernel size must be less than or equal to 7"
        sigma = kernel_kwargs.get('sigma', 0.3*((size-1)*0.5-1)+0.8)  # 自动计算sigma
        kernel = gaussion_kernel(size, sigma) @ gaussion_kernel(size, sigma).T
    elif kernel_type == 'lanczos':
        a = kernel_kwargs.get('a', 3)
        axis = np.linspace(-(size//2), size//2, size)
        xx, yy = np.meshgrid(axis, axis)
        kernel = np.sinc(xx) * np.sinc(xx/a) * np.sinc(yy) * np.sinc(yy/a)
    else:
        print(kernel_type)
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    # assure normalization
    kernel = kernel / np.sum(kernel)

    return kernel

def conv2d_downsample(image, scale_factor, kernel):
    """ 
    param: image: np.ndarray
    param: scale_factor: float, must be greater than 1
    param: kernel: np.ndarray, kernel to apply on the image
    """
    assert scale_factor > 1 , "scale factor must be greater than 1"
    # width, height, channels
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    new_w, new_h = int(w/scale_factor), int(h/scale_factor)
    # print(f"{h}x{w} -> {new_h}x{new_w}")
    if c == 1:
        new_image = np.zeros((new_h, new_w), dtype=image.dtype)
    else:
        new_image = np.zeros((new_h, new_w, c), dtype=image.dtype)
    
    # kernal dimensions
    kh, kw = kernel.shape[:2]
    # repeat 2d kernel to match the number of channels
    if len(kernel.shape) == 2:
        kernel = np.repeat(kernel[..., np.newaxis], c, axis=2)
    kh, kw, kc = kernel.shape

    for new_y in range(new_h):
        for new_x in range(new_w):
            # calculate the region of interest
            h_start = max(0, int(np.floor(new_y * scale_factor)))
            h_end = min(int(np.ceil((new_y + 1) * scale_factor)), h - 1)
            w_start = max(0, int(np.floor(new_x * scale_factor)))
            w_end = min(int(np.ceil((new_x + 1) * scale_factor)), w - 1)

            patch = image[h_start:h_end, w_start:w_end, :]

            # boundary check
            if patch.shape[0] < kh or patch.shape[1] < kw:
                pad_h = kh - patch.shape[0]
                pad_w = kw - patch.shape[1]
                patch = np.pad(patch, 
                              ((0, pad_h), (0, pad_w), (0, 0)), 
                              mode='reflect')  # 底部/右侧镜像
            
            # apply convolution 
            new_image[new_y, new_x] = np.sum(patch * kernel, axis=(0, 1))
            
    if new_image.dtype == np.float32:
        return np.clip(new_image, 0, 1)
