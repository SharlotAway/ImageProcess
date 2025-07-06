import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def mean_filter(image, kernel_size=3):
    """空间域均值滤波器"""
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size, c]
                result[i, j, c] = np.mean(region)
    return result

def gaussian_filter(image, kernel_size=3, sigma=None):
    """高斯滤波器"""
    if sigma is None:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    # 生成高斯核
    x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    kernel = g / g.sum()
    
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size, c]
                result[i, j, c] = np.sum(region * kernel)
    return result

def sinc_filter(image, kernel_size=7, a=3):
    """sinc函数近似滤波器（低通）"""
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # 生成sinc核
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            if x == 0 and y == 0:
                kernel[i, j] = 1.0
            else:
                r = np.sqrt(x**2 + y**2)
                kernel[i, j] = np.sinc(r / a) * np.sinc(r)
    
    # 归一化
    kernel /= np.sum(kernel)
    
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size, c]
                result[i, j, c] = np.sum(region * kernel)
    return result

def lanczos_filter(image, kernel_size=7, a=3):
    """Lanczos窗口滤波器"""
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # 生成Lanczos核
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            if x == 0 and y == 0:
                kernel[i, j] = 1.0
            elif abs(x) < a and abs(y) < a:
                kernel[i, j] = (np.sinc(x) * np.sinc(x/a)) * (np.sinc(y) * np.sinc(y/a))
            else:
                kernel[i, j] = 0.0
    
    # 归一化
    kernel /= np.sum(kernel)
    
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size, c]
                result[i, j, c] = np.sum(region * kernel)
    return result

def kaiser_filter(image, kernel_size=7, beta=6.0):
    """Kaiser窗口滤波器"""
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # 生成Kaiser窗口
    x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
    r = np.sqrt(x**2 + y**2) / (kernel_size // 2)
    mask = r <= 1.0
    
    # 计算Kaiser窗口
    from scipy.special import i0  # 零阶修正贝塞尔函数
    kaiser = np.zeros_like(r)
    for i in range(kernel_size):
        for j in range(kernel_size):
            if mask[i, j]:
                kaiser[i, j] = i0(beta * np.sqrt(1 - r[i, j]**2)) / i0(beta)
    
    # 归一化
    kaiser /= np.sum(kaiser)
    
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size, c]
                result[i, j, c] = np.sum(region * kaiser)
    return result