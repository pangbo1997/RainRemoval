import torch
from . import networks
from .base_model import BaseModel
import itertools
from .SSIM import SSIM
import torch.nn as nn
from .rpdnet import RPDNet
class L0Loss(nn.Module):
    def __init__(self,total_epoch):
        super(L0Loss,self).__init__()
        self.total_epoch=total_epoch
    def forward(self, input,output,epoch):
        return torch.mean(torch.pow(torch.abs(input-output)+1e-8,(1-epoch/self.total_epoch)*2))
class RpdnetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):


        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser
    def __init__(self, opt):

        BaseModel.__init__(self, opt)


        self.loss_names=['loss']
        self.visual_names = ['l','h','g']
        self.model_names = ['Rpd']

        self.netRpd = networks.init_net(RPDNet(),gpu_ids=opt.gpu_ids)

     

        self.loss_type=opt.loss_type
        if opt.loss_type=='ssim':
            self.criterion=SSIM()
        elif  opt.loss_type=='l1':
            self.criterion=torch.nn.L1Loss()
        elif opt.loss_type == 'l2':
            self.criterion=torch.nn.MSELoss()
        elif opt.loss_type == 'l0':
            self.criterion=L0Loss(opt.niter+opt.niter_decay-1)


        self.optimizer = torch.optim.Adam(self.netRpd.parameters(), lr=opt.lr, betas=(0.0, 0.999))


        self.optimizers.append(self.optimizer)

    def set_input(self, input):


        self.l = input[0].to(self.device)
        self.h = input[1].to(self.device)



    def forward(self):
        self.g_list=self.netRpd.forward(self.l)
        self.g=self.g_list[-1]


    def backward(self):
        if self.loss_type=='ssim':
            loss_list = [-self.criterion(self.h, g) for g in self.g_list]
        elif  self.loss_type=='l0':
            loss_list = [self.criterion(self.h, g,self.current_epoch) for g in self.g_list]
        else:
            loss_list = [self.criterion(self.h, g) for g in self.g_list]

        self.loss_loss=loss_list[-1]    
 
        loss_val=sum(loss_list)
        loss_val.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        self.set_requires_grad(self.netRpd, True) # enable backprop for D
        self.optimizer.zero_grad()  # set D's gradients to zero
        self.backward()  # calculate gradients for D
        self.optimizer.step()
