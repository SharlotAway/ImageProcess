import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
from .datasets import TrainDataset, EvalDataset

class TransposedConvNet(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale = scale_factor
        
        # 根据上采样倍数选择层数
        if scale_factor == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
            )
        elif scale_factor == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
            )
        elif scale_factor == 8:
            self.upsample = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
            )

    def forward(self, x):
        return torch.clamp(self.upsample(x), 0, 1) 
    
def train_tcn(
        scale_factor, 
        train_file,    
        valid_file,    
        epochs=50, 
        batch_size=16,
        patch_size=12, 
        save_path="model_x4.pth",
        lr=1e-3
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = TrainDataset(train_file, patch_size=patch_size, scale=scale_factor)
    valid_dataset = EvalDataset(valid_file)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1
    )
    
    model = TransposedConvNet(scale_factor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        # verbose=True
    )

    best_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 10
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        valid_loss = 0.0
        psnr_sum = 0.0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(valid_loader, desc="Evaluating"):
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                outputs = model(lr_imgs)
                loss = criterion(outputs, hr_imgs)
                valid_loss += loss.item()

                mse = ((outputs - hr_imgs) ** 2).mean()
                psnr = 10 * torch.log10(1.0 / mse)
                psnr_sum += psnr.item()
        
        valid_loss /= len(valid_loader)
        avg_psnr = psnr_sum / len(valid_loader)
        

        print(f"Epoch [{epoch+1}/{epochs}]:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, PSNR: {avg_psnr:.2f}dB")
        
        scheduler.step(valid_loss)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

    
    return model

def evaluate_tcn(model_path, eval_file, scale_factor):
    """
    使用h5格式数据评估模型
    Args:
        model_path: 模型权重文件路径
        eval_file: h5格式的评估数据集路径 
        scale_factor: 超分辨率倍数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = TransposedConvNet(scale_factor).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 使用EvalDataset加载h5数据
    eval_dataset = EvalDataset(eval_file)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=1)
    
    criterion = nn.MSELoss()
    total_psnr = 0
    total_loss = 0
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(eval_loader, desc="Evaluating"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            sr_imgs = model(lr_imgs)

            loss = criterion(sr_imgs, hr_imgs)
            mse = ((sr_imgs - hr_imgs) ** 2).mean()
            psnr = 10 * torch.log10(1.0 / mse)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
    
    avg_loss = total_loss / len(eval_loader)
    avg_psnr = total_psnr / len(eval_loader)
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f}dB")
    
    return avg_loss, avg_psnr

if __name__ == "__main__":
    scale_factor = 2
    model = train_tcn(
        scale_factor=scale_factor,
        train_file="./data/DIV2K_x4_valid.h5",  
        valid_file="./data/DIV2K_x4_valid.h5", 
        epochs=200,
        batch_size=16,
        patch_size=12,
        save_path="checkpoints/best_tcn.pth",
        lr=2e-3
    )