import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def nearest_neighbor_upsample(image, scale_factor):
    """最近邻上采样
    Args:
        image: 输入图像数组 [0,1]范围的float32
        scale_factor: 放大倍数（可以是非整数）
    Returns:
        np.ndarray: 上采样后的图像数组，值范围[0,1]
    """
    assert scale_factor > 1, "scale factor must be greater than 1"
    
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    # print(f"Upsampling: {h}x{w} -> {new_h}x{new_w}")
    
    if c == 1:
        new_image = np.zeros((new_h, new_w), dtype=np.float32)
    else:
        new_image = np.zeros((new_h, new_w, c), dtype=np.float32)
    
    for new_y in range(new_h):
        for new_x in range(new_w):
            # 计算对应的原图坐标（使用向下取整）
            src_x = int(new_x / scale_factor)
            src_y = int(new_y / scale_factor)
            
            # 确保不超出原图范围
            src_x = min(src_x, w - 1)
            src_y = min(src_y, h - 1)
            
            # 复制像素值
            new_image[new_y, new_x] = image[src_y, src_x]
    
    return new_image

def bilinear_interpolation_upsample(image, scale_factor):
    """双线性插值上采样
    Args:
        image: 输入图像数组 [0,1]范围的float32
        scale_factor: 放大倍数（可以是非整数）
    Returns:
        np.ndarray: 上采样后的图像数组，值范围[0,1]
    """
    assert scale_factor > 1, "scale factor must be greater than 1"
    
    # 获取原始图像尺寸
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    
    # 计算新图像尺寸
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    # print(f"Upsampling: {h}x{w} -> {new_h}x{new_w}")
    
    # 创建新图像数组
    if c == 1:
        new_image = np.zeros((new_h, new_w), dtype=np.float32)
    else:
        new_image = np.zeros((new_h, new_w, c), dtype=np.float32)
    
    # 双线性插值
    for new_y in range(new_h):
        for new_x in range(new_w):
            # 计算在原图中的精确位置
            src_x = new_x / scale_factor
            src_y = new_y / scale_factor
            
            # 获取四个最近的像素点坐标
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, w - 1)
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, h - 1)
            
            # 计算插值权重
            wx = src_x - x0
            wy = src_y - y0
            
            # 双线性插值计算
            if c == 1:
                new_image[new_y, new_x] = (
                    image[y0, x0] * (1 - wx) * (1 - wy) +
                    image[y0, x1] * wx * (1 - wy) +
                    image[y1, x0] * (1 - wx) * wy +
                    image[y1, x1] * wx * wy
                )
            else:
                new_image[new_y, new_x] = (
                    image[y0, x0, :] * (1 - wx) * (1 - wy) +
                    image[y0, x1, :] * wx * (1 - wy) +
                    image[y1, x0, :] * (1 - wx) * wy +
                    image[y1, x1, :] * wx * wy
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


def bicubic_interpolation_upsample(image, scale_factor):
    """
    Upsample an image using bicubic interpolation.
    
    Parameters:
    image (np.ndarray): Input image with shape (height, width) or (height, width, channels)
    scale_factor (float): Upsampling factor (must be > 1)
    
    Returns:
    np.ndarray: Upsampled image
    """
    assert scale_factor > 1, "scale factor must be greater than 1"
    
    # Get image dimensions
    h, w = image.shape[:2]
    c = image.shape[2] if len(image.shape) == 3 else 1
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Initialize output image
    if c == 1:
        new_image = np.zeros((new_h, new_w), dtype=image.dtype)
    else:
        new_image = np.zeros((new_h, new_w, c), dtype=image.dtype)
    
    # Precompute x and y positions in the original image space
    x_positions = np.linspace(0, w - 1, new_w)
    y_positions = np.linspace(0, h - 1, new_h)
    
    for new_y in range(new_h):
        y = y_positions[new_y]
        y_floor = int(np.floor(y))
        dy = y - y_floor
        
        # Compute vertical kernel weights for 4x4 neighborhood
        y_weights = np.zeros(4)
        for ky in range(-1, 3):
            y_idx = y_floor + ky
            if 0 <= y_idx < h:
                y_weights[ky + 1] = bicubic_kernel(dy - ky)
        
        for new_x in range(new_w):
            x = x_positions[new_x]
            x_floor = int(np.floor(x))
            dx = x - x_floor
            
            # Compute horizontal kernel weights for 4x4 neighborhood
            x_weights = np.zeros(4)
            for kx in range(-1, 3):
                x_idx = x_floor + kx
                if 0 <= x_idx < w:
                    x_weights[kx + 1] = bicubic_kernel(dx - kx)
            
            # Initialize result
            result = np.zeros(c) if c > 1 else 0
            
            # Compute bicubic interpolation using 4x4 neighborhood
            for ky in range(-1, 3):
                y_idx = y_floor + ky
                if y_idx < 0 or y_idx >= h:
                    continue
                
                for kx in range(-1, 3):
                    x_idx = x_floor + kx
                    if x_idx < 0 or x_idx >= w:
                        continue
                    
                    weight = y_weights[ky + 1] * x_weights[kx + 1]
                    
                    # Handle both single-channel and multi-channel images
                    if c > 1:
                        result += image[y_idx, x_idx] * weight
                    else:
                        result += image[y_idx, x_idx] * weight
            
            new_image[new_y, new_x] = result
    
    return new_image


def pixel_shuffle_upsample(input, scale_factor):
    assert scale_factor > 1 , "scale factor must be greater than 1"
    h,w,c = input.shape
    r = int(scale_factor)

    # Verify that channels can be divided by s²
    if c % (r * r) != 0:
        return None
        # raise ValueError(f"Input channels ({c}) must be divisible by the square of scale factor ({r*r})")

    out_c = c // (r * r) 

    reshaped = input.reshape(h, w, r, r, out_c)
    transposed = np.transpose(reshaped, (0, 2, 1, 3, 4))
    new_image = transposed.reshape(h * r, w * r, out_c)
    return new_image
