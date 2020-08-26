# Single Image Deraining via Scale-space Invariant Attention Neural Network
The paper[[Link]](https://arxiv.org/abs/2006.05049) is accepted by ACMMM2020(oral). This is pytorch implementation for our paper.
The code is based on the [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

### Prerequisites

- python 3.6 
- pytorch 1.1.0
- opencv-python
- visdom 
###  Pretrained Models
You could directly find our pretrained models in the directory.

### Usage
To train the model, please run

    python train.py --dataroot [your dataset path] --name rpdnet_lstm_100l --model rpdnet --gpu_ids 3 
    --batch_size 18 --output_nc 3 --input_nc 3 --niter 100 --niter_decay 0 --save_epoch_freq 10
    --display_freq 40 --dataset_mode [rain100l|rain100h|rain1400] --display_freq 20 --lr 2e-4
    --color_space 'rgb' --no_html --display_env lstm
    
To generate images, please run   
    
    python eval.py --dataset_path [your dataset path] --model_path [your model path] 
    --save_path [your save path] --gpu_id 0
    

### Performance
The evaluation metrics are provided by [Ren](https://github.com/csdwren/PReNet).
The performances on the four  datasets are listed below:



| Dataset| PSNR |SSIM | 
| :-----:| :-----: | :-----:| 
| Rain100L |   38.80  | 0.984 |
| Rain100H   |   30.33  | 0.909 | 
| Rain1400    |  32.80 | 0.946 | 
| Rain12    |   37.42 | 0.967 | 
 

