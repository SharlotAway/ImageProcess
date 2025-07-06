import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt
import torch

def load_image(image_path, scale_factor=2, adjust=True):
    image = Image.open(image_path).convert('RGB')
    # 添加 scale factor 是为了可是适配下采样后上采样
    if adjust:
        image_width = (image.width // scale_factor) * scale_factor
        image_height = (image.height // scale_factor) * scale_factor
        image = image.resize((image_width, image_height), resample=Image.BICUBIC)
    np_img = np.array(image) / 255
    return np_img.astype('float32')

def show(images, titles=None):
    if not images is None:
        pass 
    
    if isinstance(images, np.ndarray):
        images = [images]
    num_img = len(images)
    if isinstance(titles, str):
        titles = [titles] * num_img
    if titles is not None:
        assert len(titles) == len(images), 'Titles length must match images length' 

    fig, axes = plt.subplots(1, num_img, figsize=(3 * num_img, 4))
    if num_img == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        
        if titles is not None:
            ax.set_title(titles[i])
    plt.tight_layout() 
    plt.show()

def calculate_mse(img1, img2):
    """计算两幅图像的均方误差(MSE)"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have same dimensions")
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

# 0 ~ inf
def calculate_psnr(img1, img2, max_val=1.0):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float("inf")
    else:
        return 20 * np.log10(max_val) - 10 * np.log10(mse)

def gaussian_kernel_2d(size: int, sigma: float):
    """生成归一化的二维高斯核（空间域）"""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """对 2D 图像进行边缘填充的 2D 卷积（单通道）"""
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            result[i, j] = np.sum(region * kernel)
    return result

def calculate_ssim(img1: np.ndarray, img2: np.ndarray, size: int = 11, sigma: float = 1.5, data_range=1.0) -> float:
    """计算 SSIM 指标，支持灰度或 RGB 图像，使用纯 NumPy"""
    assert img1.shape == img2.shape, "图像尺寸不匹配"
    if img1.ndim == 3 and img1.shape[2] == 3:
        # 对 RGB 每个通道分别计算 SSIM，再平均
        return np.mean([calculate_ssim(img1[..., i], img2[..., i], size, sigma, data_range) for i in range(3)])

    # channel wise
    kernel = gaussian_kernel_2d(size, sigma)

    mu1 = conv2d(img1, kernel)
    mu2 = conv2d(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = conv2d(img2 ** 2, kernel) - mu2_sq
    sigma12   = conv2d(img1 * img2, kernel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))

## pip install lpips
# 需要安装：pip install torch lpips
def calculate_lpips(img1: np.ndarray, img2: np.ndarray, net='alex') -> float:
    import torch
    import lpips

    # Create model once
    global lpips_model
    if 'lpips_model' not in globals():
        lpips_model = lpips.LPIPS(net=net)
        lpips_model.eval()

    def preprocess(img):
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return img * 2 - 1  # to [-1, 1]

    img1_t = preprocess(img1)
    img2_t = preprocess(img2)

    with torch.no_grad():
        dist = lpips_model(img1_t, img2_t).item()
    return dist
