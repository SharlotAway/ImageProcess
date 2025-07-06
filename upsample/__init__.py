import torch 
import torch.backends.cudnn as cudnn 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt

from .interpolation import(
    nearest_neighbor_upsample,
    bilinear_interpolation_upsample,
    bicubic_interpolation_upsample,
    pixel_shuffle_upsample
)

from .utils import *
from .ESPCN.models import ESPCN
from .RDN.models import RDN
from .SRCNN.models import SRCNN 
from .TCN.models import TransposedConvNet

def tcn_upsampling(img, scale_factor, ckpt_path, device='cuda'):
    model = TransposedConvNet(scale_factor).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    lr_img = torch.from_numpy(img.transpose((2,0,1))).to(device)
    with torch.no_grad():
        preds = model(lr_img)
    new_image = preds.detach().cpu().numpy().transpose((1,2,0))
    return new_image

def espcn_upsampling(image, scale_factor, ckpt_path, device='cuda', bicubic_hr_img=None):
    model = ESPCN(scale_factor).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    if bicubic_hr_img == None:
        bicubic_hr_img = bicubic_interpolation_upsample(image, scale_factor)
    lr_tensor, _ = preprocess(image * 255, device)
    _, ycbcr = preprocess(bicubic_hr_img * 255, device)

    with torch.no_grad():
        preds = model(lr_tensor).clamp(0.0, 1.0)
    
    preds = preds.mul(255.0).cpu().numpy().squeeze()
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    # show(output, [output.shape])
    new_image = (output / 255.0).astype('float32')
    return new_image

def rdn_upsampling(
    image, scale_factor, ckpt_path, 
    device='cuda', 
    num_features: int = 64,
    growth_rate: int = 64,
    num_blocks: int = 16,
    num_layers: int = 8
):
    model = RDN(
        scale_factor=scale_factor,
        num_channels=image.shape[-1],
        num_features=num_features,
        growth_rate=growth_rate,
        num_blocks=num_blocks,
        num_layers=num_layers
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    lr_tensor = torch.from_numpy(image.transpose([2,0,1])).to(device).unsqueeze(0)

    with torch.no_grad():
        preds = model(lr_tensor).squeeze(0)
    
    new_image = preds.detach().cpu().numpy().transpose([1,2,0])
    return new_image


def srcnn_upsampling(
    image, scale_factor, ckpt_path, device='cuda', bicubic_hr_img=None
):
    model = SRCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    
    lr_img = image * 255
    if bicubic_hr_img is None:
        bicubic_hr_img = bicubic_interpolation_upsample(image, scale_factor)
    ycbcr = convert_rgb_to_ycbcr(bicubic_hr_img)
    
    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    # return preds
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    # new_image = (output / 255.0).astype('float32')
    new_image = output
    return new_image

def rule_upsample(image, scale_factor, us_name=None):
    image = image.astype(np.float32)
    assert us_name.lower() in ['neighbor', 'bilinear', 'bicubic', 'shuffle'], f"Unknown usampling method {us_name}."
    upsample_dict = {
        "neighbor": nearest_neighbor_upsample,
        "bilinear": bilinear_interpolation_upsample,
        "bicubic": bicubic_interpolation_upsample,
        "shuffle": pixel_shuffle_upsample,
    }
    return upsample_dict[us_name](image, scale_factor)
        

def learnable_upsample(image, scale_factor, us_name=None, ckpt_path=None, device='cuda'):
    upsample_models = ['TCN', 'ESPCN', "RDN", "SRCNN"]
    assert us_name in upsample_models
    ckpt_dict = {
        'TCN': "./upsample/TCN/checkpoints/best_tcn.pth",
        'ESPCN': "./upsample/ESPCN/checkpoints/x2/best.pth",
        'RDN': "./upsample/RDN/checkpoints/x2/best.pth",
        'SRCNN': "./upsample/SRCNN/checkpoints/x2/best.pth"
    }
        
    upsample_dict = {
        'ESPCN': espcn_upsampling,
        'RDN': rdn_upsampling,
        'TCN': tcn_upsampling,
        'SRCNN': srcnn_upsampling
    }
    if ckpt_path is None:
        ckpt_path = ckpt_dict[us_name]

    return upsample_dict[us_name](image, scale_factor, ckpt_path)

def upsample(image, scale_factor, us_name=None, ckpt_path=None, device='cuda'):
    upsample_names = ['neighbor', 'bilinear', 'bicubic', 'shuffle']
    upsample_models = ['TCN', 'ESPCN', "RDN", "SRCNN"]
    if us_name.lower() in upsample_names:
        return rule_upsample(image, scale_factor, us_name)
    elif us_name in upsample_models:
        return learnable_upsample(image, scale_factor, us_name, ckpt_path)
    else:
        raise KeyError(f"Unkown upsampling method {us_name}")