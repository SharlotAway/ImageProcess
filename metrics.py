import torch
import torch.nn.functional as F
import lpips
import numpy as np
import time
from tabulate import tabulate
import csv

def print_metrics(metrics_dict):
    """
    将指标字典转换为表格形式输出
    
    参数:
        metrics_dict: 包含各种方法指标的字典，格式如您提供的示例
    """
    headers = ["Method", "MSE", "PSNR (dB)", "SSIM", "LPIPS"]
    
    table_data = []
    for method, metrics in metrics_dict.items():
        row = [
            method.upper(),  
            f"{metrics['mse']:.2e}",  
            f"{metrics['psnr']:.2f}",  
            f"{metrics['ssim']:.6f}",  
            f"{metrics['lpips']:.6f}"  
        ]
        table_data.append(row)
    

    table_data.sort(key=lambda x: float(x[2]), reverse=False)
    
    print(tabulate(table_data, headers=headers, tablefmt="latex"))
    print("\nTable has bee sorted by psnr.")


class ImageMetricsCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        
        # 初始化LPIPS模型
        self.lpips_model = lpips.LPIPS(net='alex', spatial=False).to(device)
        self.lpips_model.eval()
    
    def calculate_all(self, ref_img, gen_img, data_range=255.0):
        """
        一次性计算所有指标
        :param ref_img: 参考图像 (tensor或numpy)
        :param gen_img: 生成图像 (tensor或numpy)
        :param data_range: 数据的最大值
        :return: 包含所有指标的字典
        """
        # 转换输入为torch tensor
        if isinstance(ref_img, np.ndarray):
            ref_tensor = torch.from_numpy(ref_img.transpose(2, 0, 1)).unsqueeze(0)
            gen_tensor = torch.from_numpy(gen_img.transpose(2, 0, 1)).unsqueeze(0)
            
            # 转换为0-1范围
            ref_tensor = ref_tensor.float() / 255.0
            gen_tensor = gen_tensor.float() / 255.0
            data_range = 1.0
        else:
            ref_tensor = ref_img.clone()
            gen_tensor = gen_img.clone()
        
        # 转换到正确设备
        ref_tensor = ref_tensor.to(self.device)
        gen_tensor = gen_tensor.to(self.device)
        
        metrics = {}
        
        # 计算MSE和PSNR
        metrics['mse'] = self.calculate_mse(ref_tensor, gen_tensor, data_range)
        metrics['psnr'] = self.calculate_psnr(metrics['mse'], data_range)
        
        # 计算SSIM
        metrics['ssim'] = self.calculate_ssim(ref_tensor, gen_tensor, data_range)
        
        # 计算LPIPS
        metrics['lpips'] = self.calculate_lpips(ref_tensor, gen_tensor)
        
        return metrics
    
    def calculate_mse(self, ref_img, gen_img, data_range=255.0):
        """计算MSE"""
        # 统一数据类型
        ref_img = ref_img.float()
        gen_img = gen_img.float()
        
        # 如果数据范围大于1，转换为0-1
        if data_range > 1:
            ref_img = ref_img / data_range
            gen_img = gen_img / data_range
        
        # 计算差值
        diff = ref_img - gen_img
        
        # 计算MSE
        mse = torch.mean(diff**2)
        return mse.item()
    
    def calculate_psnr(self, mse, data_range=255.0):
        """计算PSNR"""
        mse = torch.tensor(mse)
        # 如果数据范围大于1，表示实际范围是0-data_range，PSNR需要调整
        psnr = 10 * torch.log10((data_range**2) / mse)
        return psnr.item()
    
    def calculate_ssim(self, ref_img, gen_img, data_range=255.0, window_size=11):
        """计算SSIM - 基于PyTorch实现"""
        # 如果数据范围大于1，转换为0-1
        if data_range > 1:
            ref_img = ref_img / data_range
            gen_img = gen_img / data_range
            data_range = 1.0
        
        # 确定通道数
        channels = ref_img.shape[1]
        
        # 创建高斯核
        sigma = 1.5  # 默认值
        gaussian_kernel = self._create_gaussian_kernel(window_size, sigma)
        gaussian_kernel = gaussian_kernel.to(ref_img.device)
        
        # 初始化结果
        ssim_value = 0.0
        
        # 分别计算每个通道的SSIM
        for c in range(channels):
            channel_ref = ref_img[:, c:c+1, :, :]
            channel_gen = gen_img[:, c:c+1, :, :]
            
            # 计算均值
            mu1 = F.conv2d(channel_ref, gaussian_kernel, padding=window_size//2)
            mu2 = F.conv2d(channel_gen, gaussian_kernel, padding=window_size//2)
            
            # 计算方差和协方差
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(channel_ref*channel_ref, gaussian_kernel, padding=window_size//2) - mu1_sq
            sigma2_sq = F.conv2d(channel_gen*channel_gen, gaussian_kernel, padding=window_size//2) - mu2_sq
            sigma12 = F.conv2d(channel_ref*channel_gen, gaussian_kernel, padding=window_size//2) - mu1_mu2
            
            # SSIM公式
            C1 = (0.01 * data_range) ** 2
            C2 = (0.03 * data_range) ** 2
            
            ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            ssim_value += torch.mean(ssim_map)
        
        return ssim_value.item() / channels
    
    def calculate_lpips(self, ref_img, gen_img):
        """计算LPIPS"""
        # LPIPS输入需要在[-1,1]范围内
        normalized_ref = (ref_img * 2) - 1
        normalized_gen = (gen_img * 2) - 1
        
        with torch.no_grad():
            lpips_score = self.lpips_model(normalized_ref, normalized_gen)
        
        return lpips_score.item()
    
    def _create_gaussian_kernel(self, window_size, sigma):
        """创建高斯核"""
        def gaussian(x, mu, sigma):
            return (1.0 / (sigma * np.sqrt(2*np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        kernel = np.zeros((1, 1, window_size, window_size))
        center = window_size // 2
        for i in range(window_size):
            for j in range(window_size):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                kernel[0, 0, i, j] = gaussian(distance, 0, sigma)
        
        # 归一化
        kernel /= kernel.sum()
        return torch.tensor(kernel, dtype=torch.float32)

def write_exp_results(results, csv_path):
    # 准备CSV表头和行数据
    headers = ["Method"] + list(next(iter(results.values())).keys())
    rows = []
    for method, metrics in results.items():
        row = [method] + list(metrics.values())
        rows.append(row)

    # 写入CSV文件
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # 写入表头
        writer.writerows(rows)    # 写入数据行

    print("Successfully restore experiment results.")