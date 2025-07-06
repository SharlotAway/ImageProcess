import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import matplotlib.pyplot as plt

from models import RDN
from utils import convert_rgb_to_y, denormalize, calc_psnr

def show(images, titles=None, cmap='gray'):
    import numpy as np
    if isinstance(images, np.ndarray):
        images = [images]
    num_img = len(images)
    if isinstance(titles, str):
        titles = [titles] * num_img
    if titles is not None:
        assert len(titles) == len(images), 'Titles length must match images length'
    fig, axes = plt.subplots(1, num_img, figsize=(4 * num_img, 4))
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
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-blocks', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = RDN(scale_factor=args.scale,
                num_channels=3,
                num_features=args.num_features,
                growth_rate=args.growth_rate,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    # 图像预处理
    image = pil_image.open(args.image_file).convert('RGB')
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)

    lr_tensor = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    hr_tensor = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    lr_tensor = torch.from_numpy(lr_tensor).to(device)
    hr_tensor = torch.from_numpy(hr_tensor).to(device)

    # 推理
    with torch.no_grad():
        preds = model(lr_tensor).squeeze(0)

    preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
    hr_y = convert_rgb_to_y(denormalize(hr_tensor.squeeze(0)), dim_order='chw')
    preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
    hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

    psnr = calc_psnr(hr_y, preds_y)
    print('PSNR: {:.2f} dB'.format(psnr))

    # 图像还原为RGB可视化格式
    output_img = denormalize(preds).permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    hr_img = np.array(hr)
    bicubic_img = np.array(bicubic)
    input_img = np.array(lr.resize((hr.width, hr.height), resample=pil_image.BICUBIC))  # 模糊输入图

    # 可视化原图、bicubic、RDN输出
    show([hr_img, bicubic_img, output_img],
         titles=['Original HR', f'Bicubic x{args.scale}', f'RDN x{args.scale}'])