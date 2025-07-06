import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt

from models import ESPCN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr

def show(images, titles=None, cmap='gray'):
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
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        if titles is not None:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ESPCN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    # 加载并调整图像
    image = pil_image.open(args.image_file).convert('RGB')
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr_img = image.resize((image_width, image_height), resample=pil_image.BICUBIC)  # 原始高分图
    lr_img = hr_img.resize((hr_img.width // args.scale, hr_img.height // args.scale), resample=pil_image.BICUBIC)  # 下采样
    bicubic_img = lr_img.resize((lr_img.width * args.scale, lr_img.height * args.scale), resample=pil_image.BICUBIC)  # Bicubic 插值放大
    bicubic_img.save(args.image_file.replace('.', f'_bicubic_x{args.scale}.'))

    # 转 Tensor 做模型推理
    lr_tensor, _ = preprocess(lr_img, device)
    hr_tensor, _ = preprocess(hr_img, device)
    _, ycbcr = preprocess(bicubic_img, device)  # 提取 CbCr 分量

    # 模型超分
    with torch.no_grad():
        preds = model(lr_tensor).clamp(0.0, 1.0)

    # PSNR
    psnr = calc_psnr(hr_tensor, preds)
    print('PSNR: {:.2f}'.format(psnr))

    # 处理输出为图像格式
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    espcn_img = pil_image.fromarray(output)
    espcn_img.save(args.image_file.replace('.', f'_espcn_x{args.scale}.'))

    # 可视化：4图对比
    lr_np = np.array(lr_img)
    hr_np = np.array(hr_img)
    bicubic_np = np.array(bicubic_img)
    espcn_np = np.array(espcn_img)

    show(
        [lr_np, hr_np, bicubic_np, espcn_np],
        titles=[
            f'LR Input ({lr_np.shape[1]}×{lr_np.shape[0]})',
            f'Original HR ({hr_np.shape[1]}×{hr_np.shape[0]})',
            f'Bicubic x{args.scale}',
            f'ESPCN x{args.scale}'
        ],
        cmap=None
    )