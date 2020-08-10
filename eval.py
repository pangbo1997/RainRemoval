import torch
import os

import numpy as np

import os
from models.rpdnet import RPDNet
import argparse
import cv2
parser=argparse.ArgumentParser()
parser.add_argument('--dataset_path',type=str)
parser.add_argument('--model_path',type=str)
parser.add_argument('--gpu_id',type=int,default=0)
parser.add_argument('--save_path',type=str)
opt=parser.parse_args()

def main():

    os.makedirs(opt.save_path,exist_ok=True)
    model = RPDNet()
    model.cuda(opt.gpu_id)
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()
    dir=os.listdir(opt.dataset_path)

    for img_name in dir:
        print('process: '+img_name)
        img = cv2.imread(os.path.join(opt.dataset_path,img_name))
        b, g, r = cv2.split(img)

        img = cv2.merge([r, g, b])

        img = (np.float32(img))/255.0
        img = np.expand_dims(img.transpose(2, 0, 1), 0)

        img = torch.tensor(img).cuda(opt.gpu_id)
        with torch.no_grad():
            gen_img=model(img)[-1].data.cpu().numpy()[0].transpose(1,2,0)
            b, g, r = cv2.split(255*gen_img)
            save_out = cv2.merge([r, g, b])

            cv2.imwrite(os.path.join(opt.save_path, img_name), save_out)

  



if __name__=='__main__':
    main()
