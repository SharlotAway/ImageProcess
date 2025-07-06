import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt


def nearest_neighbor_downsample(image, scale_factor):
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

    # Nearest neighbor downsampling
    for new_y in range(new_h):
        for new_x in range(new_w):
            # nearest integer coordianate in the original image
            y = min(round(new_y * scale_factor), h - 1)
            x = min(round(new_x * scale_factor), w - 1)

            new_image[new_y, new_x] = image[y, x]
    return new_image


def bilinear_interpolation_downsample(image, scale_factor):
    """
    Perform bilinear interpolation downsampling with correct weights.
    
    Parameters:
    image (np.ndarray): Input image with shape (height, width) or (height, width, channels)
    scale_factor (float): Downsampling factor (must be > 1)
    
    Returns:
    np.ndarray: Downsampled image
    """
    assert scale_factor > 1, "scale factor must be greater than 1"
    
    # Get image dimensions
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    new_w, new_h = int(w / scale_factor), int(h / scale_factor)
    
    # print(f"{h}x{w} -> {new_h}x{new_w}")
    
    # Initialize output image
    if c == 1:
        new_image = np.zeros((new_h, new_w), dtype=image.dtype)
    else:
        new_image = np.zeros((new_h, new_w, c), dtype=image.dtype)
    
    for new_x in range(new_w):
        for new_y in range(new_h):
            # Calculate position in original image
            x = new_x * scale_factor
            y = new_y * scale_factor
            
            # Get integer coordinates of 4 nearest neighbors
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
            
            # Calculate fractional parts (relative distances)
            dx = x - x0
            dy = y - y0
            
            # Compute bilinear weights
            w00 = (1 - dx) * (1 - dy)  # Top-left
            w01 = dx * (1 - dy)        # Top-right
            w10 = (1 - dx) * dy        # Bottom-left
            w11 = dx * dy              # Bottom-right
            
            # Perform weighted sum of 4 neighbors
            new_image[new_y, new_x] = (
                image[y0, x0] * w00 + 
                image[y0, x1] * w01 + 
                image[y1, x0] * w10 + 
                image[y1, x1] * w11
            )

    return new_image


def bicubic_kernel(x, a=-0.5):
    """
    Bicubic interpolation kernel function.
    See https://en.wikipedia.org/wiki/Bicubic_interpolation for details.
    """
    abs_x = np.abs(x)
    if abs_x <= 1:
        return (a + 2) * abs_x**3 - (a + 3) * abs_x**2 + 1
    elif 1 < abs_x < 2:
        return a * abs_x**3 - 5*a * abs_x**2 + 8*a * abs_x - 4*a
    else:
        return 0


def bicubic_interpolation_downsample(image, scale_factor):
    """
    Downsample an image using bicubic interpolation.
    
    Parameters:
    image (np.ndarray): Input image with shape (height, width) or (height, width, channels)
    scale_factor (float): Downsampling factor (must be > 1)
    
    Returns:
    np.ndarray: Downsampled image
    """
    assert scale_factor > 1, "scale factor must be greater than 1"
    
    # Get image dimensions
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    new_h, new_w = int(h / scale_factor), int(w / scale_factor)
    
    # print(f"{h}x{w} -> {new_h}x{new_w}")
    
    # Initialize output image
    if c == 1:
        new_image = np.zeros((new_h, new_w), dtype=image.dtype)
    else:
        new_image = np.zeros((new_h, new_w, c), dtype=image.dtype)
    
    # Precompute x and y positions in the original image
    x_positions = np.linspace(0, w - 1, new_w)
    y_positions = np.linspace(0, h - 1, new_h)
    
    for new_y in range(new_h):
        y = y_positions[new_y]
        y_floor = int(y)
        dy = y - y_floor
        
        # Compute vertical kernel weights for 4x4 neighborhood
        y_weights = np.zeros(4)
        for ky in range(-1, 3):
            if 0 <= y_floor + ky < h:
                y_weights[ky + 1] = bicubic_kernel(dy - ky)
        
        for new_x in range(new_w):
            x = x_positions[new_x]
            x_floor = int(x)
            dx = x - x_floor
            
            # Compute horizontal kernel weights for 4x4 neighborhood
            x_weights = np.zeros(4)
            for kx in range(-1, 3):
                if 0 <= x_floor + kx < w:
                    x_weights[kx + 1] = bicubic_kernel(dx - kx)
            
            # Initialize result
            result = np.zeros(c) if c > 1 else 0
            
            # Compute bicubic interpolation
            for ky in range(-1, 3):
                y_idx = y_floor + ky
                if y_idx < 0 or y_idx >= h:
                    continue
                
                for kx in range(-1, 3):
                    x_idx = x_floor + kx
                    if x_idx < 0 or x_idx >= w:
                        continue
                    
                    weight = y_weights[ky + 1] * x_weights[kx + 1]
                    result += image[y_idx, x_idx] * weight
            
            new_image[new_y, new_x] = result
    
    return new_image


def pixel_unshuffle_downsample(image, scale_factor, return_list=True):
    """
    使用 PIL 和 numpy 实现 Pixel Unshuffle 下采样
    输入:
        image: H x W x C 的 numpy 数组
        scale_factor: int
    输出:
        下采样后的 numpy 数组，shape: (H//r, W//r, C*r*r)
    """
    assert scale_factor > 1 , "scale factor must be greater than 1"
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    h, w, c = image.shape
    r = int(scale_factor)
    if scale_factor - r != 0:
        print(f"Pixel unshuffule accept integer scale factor {r}.")
    assert h % r == 0 and w % r == 0, "Height and width must be divisible by scale_factor"
    # 重塑并转置
    new_image = image.reshape(h//r, r, w//r, r, c)
    new_image = np.transpose(new_image, (0, 2, 1, 3, 4))
    new_image = new_image.reshape(h//r, w//r, c * r * r)

    if return_list:
        new_images = []
        for i in range(r * r):
            new_images.append(new_image[:,:,i*c:(i+1)*c])
        
        return new_images
    else:
        return new_image