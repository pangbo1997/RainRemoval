# Read Me
This project can be used for mage restoration including blur effect, noise effect and etc. Specially, considering the multi-scale characteristic of the samples, we design a neural network with scale 
invariant attention mechanism.

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
    


 

