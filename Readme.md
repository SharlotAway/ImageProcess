# Down Sampling and Up Sampling with Filters

This report evaluates image upsampling and downsampling methods. Given an input image, we perform sequential 1/2 downsampling and 2× upsampling to generate a reconstructed image, aiming to identify the combination that minimizes reconstruction error. We tested classical algorithms (nearest-neighbor, bilinear, bicubic interpolation) and found bilinear downsampling + bicubic upsampling delivers minimal error across various images. Performance was evaluated using MSE, PSNR, SSIM and LPIPS metrics. Implementation used Python with, main dependencies include PIL, NumPy, and PyTorch.

## Environment
To run our code, create an environment first.

```bash
git clone git@github.com:SharlotAway/ImageProcess.git
cd ImageProcess
conda create -n sample python=3.10 -y
conda activate sample
pip install -r requirements.txt
```

## Methods
We packed 12 downsampling methods and 8 upsampling methods respectively in `downsample` and `upsample` folder, and 8 filters (3 frequency domain and 5 spatial domain) in `filters` folder. To customize these algorithms, you can open the folder to tune hyper parameters.Also you can train learanale upsampling methods with the scripts in jupyters in each folder.
`./experiment.ipynb` includes every experiments in our paper. 

## Dataset
Download DIV2K dataset from this [link](https://data.vision.ee.ethz.ch/cvl/DIV2K/) into `./dataset` folder in seperate folders. Then you can use this at wherever in the reposity by direct to the path. You can also use images in `./asserts` for simple implementations.

First go into the folder with

```bash
cd ./upsample/ESPCN # or ./upsample/RDN, ./upsample SRCNN
```

Prepare train set with

```bash
# for train set
python prepare.py --images-dir ../dataset/DIV2K/train/X4 \
  --output-path data/DIV2K_x4_train.h5 \
  --eval False
```

And Validation set with 
```python
python prepare.py --images-dir ../dataset/DIV2K/valid/X4 \
  --output-path data/DIV2K_x4_valid.h5 \
  --eval False
```

## Training
To train learnable upsampling methods, you need to download a dataset and convert the dataset into `.h5` files through `prepare.py` in the corresponding folders in `./upsample` . These methods inlude Transposed Convolutional Network (TCN), Efficient Sub-Pixel Convolutional Neural network (ESPCN, [paper link](https://arxiv.org/abs/1609.05158)), Residual Deep Neural Network (RDN, [paper link](https://arxiv.org/abs/1802.08797)), and Super-Resolution Convolutional Neural Network (SRCNN, [paper link](https://arxiv.org/abs/1501.00092)). 
==TCN==

```bash
cd ./upsample/TCN
python model.py
```

==ESPCN==
``bash
cd ./upsample/ESPCN
python train.py --train-file data/DIV2K_x4_train.h5 \
    --eval-file data/DIV2K_x4_valid.h5 \
    --outputs-dir checkpoints \
    --scale 2 \
    --lr 2e-4 \
    --batch-size 128 \
    --num-epochs 400
```
