import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def ideal_lowpass_filter(h, w, cutoff_freq):
    """创建理想低通滤波器"""
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return (distance <= cutoff_freq).astype(np.float64)

def gaussian_lowpass_filter(h, w, cutoff_freq):
    """创建高斯低通滤波器"""
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance_sq = (x - center_x)**2 + (y - center_y)**2
    return np.exp(-distance_sq / (2 * cutoff_freq**2))

def butterworth_lowpass_filter(h, w, cutoff_freq, n=2):
    """创建巴特沃斯低通滤波器"""
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return 1 / (1 + (distance / cutoff_freq)**(2*n))
