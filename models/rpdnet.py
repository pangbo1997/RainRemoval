import torch.nn as nn
import torch
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self,ch,kernel_size,stride,pad):
        super(ResBlock,self).__init__()
        self.body=nn.Sequential(*[nn.Conv2d(ch,ch,kernel_size,stride,pad),
            nn.ReLU(),
            nn.Conv2d(ch,ch,kernel_size,stride,pad),
            nn.ReLU()
            ])
    def forward(self,x):
        return F.relu(self.body(x)+x)

class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel):
        super(ConvLSTM,self).__init__()
        pad_x = int( (kernel - 1) / 2)
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x,bias=True)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x,bias=True)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x,bias=True)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x,bias=True)

        pad_h = int((kernel - 1) / 2)
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h,bias=False)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h,bias=False)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h,bias=False)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h,bias=False)


    def forward(self, x, prev):
        h,c=prev
        f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
        i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
        o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
        j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
        c = f * c + i * j
        h = o * F.tanh(c)

        return h, [h, c]

import numpy as np
def gaussian_filter(K_size,sigma):
    pad=K_size//3
    K=np.zeros((K_size,K_size),dtype=np.float32)
    for x in range(-pad,-pad+K_size):
        for y in range(-pad,-pad+K_size):
            K[y+pad,x+pad]=np.exp(-(x**2+y**2)/(2*(sigma**2)))
    K/=(2*np.pi*sigma*sigma)
    K/=K.sum()
    return K

def gaussian_blur(img_tensor,filter):
    kernel=torch.tensor(filter).unsqueeze(0).unsqueeze(0)
    kernel=np.repeat(kernel,img_tensor.shape[1],axis=0)
    kernel=nn.Parameter(data=kernel,requires_grad=False).to(img_tensor.device)
    return F.conv2d(img_tensor,kernel,padding=int(kernel.shape[-1]//2),groups=img_tensor.shape[1])
class up(nn.Module):
    def __init__(self,ch=64):
        super(up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_m= nn.Sequential(*[
                nn.Conv2d(ch*2,ch,3,1,1),
                nn.ReLU()
                ])


    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))


        x = torch.cat([x2, x1], dim=1)
        x=self.conv_m(x)
        return x

class scale_conv(nn.Module):
    def __init__(self,scale,ch=32,n_octave=3):
        super(scale_conv,self).__init__()
        self.conv_f=nn.Sequential(*[
            nn.Conv2d(ch,ch,3,1,1),
        ])
        self.conv_m=nn.Sequential(*[
            nn.Conv2d(ch*(n_octave+2),ch,3,1,1),
            nn.Sigmoid()
        ])
        self.conv_z=nn.Sequential(*[
            nn.Conv2d(ch,ch,3,1,1),
            nn.ReLU(),

        ])
        self.down_sample=nn.AvgPool2d(2**scale)
    def forward(self, x,y):
        x=self.down_sample(x)
        x=self.conv_f(x)
        x=x+x*self.conv_m(y)
        x=self.conv_z(x)
        return x

class RPDBlock(nn.Module):
    def __init__(self, input_channel,ch):
        super(RPDBlock,self).__init__()
        self.channel=ch
        self.n_scale=3
        self.n_octave=3
        sigma=1.6
        sig=np.zeros(self.n_octave+3)
        sig[0]=sigma
        k=2**(1./self.n_octave)
        for i in range(1,self.n_octave+3):
            sig_prev=(k**np.float32(i-1))*sigma
            sig_total=sig_prev*k
            sig[i]=np.sqrt(sig_total**2-sig_prev**2)

        self.filter=[gaussian_filter(3,sig[i]) for i in range(1,self.n_octave+3)]

        self.down_sample=nn.MaxPool2d(2)
        self.tran_conv=nn.Conv2d(input_channel,self.channel,3,1,1)
        self.scale_conv=nn.ModuleList([
            scale_conv(self.n_scale-i-1)
            for i in range(self.n_scale)
        ])
        self.up_conv=nn.ModuleList([
            up(ch=ch)
            for _ in range(self.n_scale-1)
        ])
        self.conv_m=nn.Sequential(*[
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU()
        ])


        self.lstm= nn.ModuleList([
            ConvLSTM(ch,ch,kernel=3)
            for _ in range(self.n_scale)
            ])



    def forward(self, x,prev):
        x=self.tran_conv(x)
        pyramid=[]
        iter_img=gaussian_blur(x,gaussian_filter(3,np.sqrt(1.6**2-0.5**2)))
        for _ in range(self.n_scale):
            octave=[iter_img]
            for i in range(self.n_octave+2):
                iter_img=gaussian_blur(iter_img,self.filter[i])
                octave.append(iter_img)
            DoG=[]
            for i in range(1,self.n_octave+3):
                DoG.append(octave[i]-octave[i-1])

            temp=DoG[0]
            for i in range(1,self.n_octave+2):
                temp=torch.cat((temp,DoG[i]),1)

            pyramid.append(temp)

            iter_img=self.down_sample(octave[self.n_octave])

        pyramid.reverse()
        for i in range(len(pyramid)):
            pyramid[i]=self.scale_conv[i](x,pyramid[i])
            pyramid[i],prev[i]=self.lstm[i](pyramid[i],prev[i])

        x=pyramid[0]
        for i in range(1,len(pyramid)):
            x=self.up_conv[i-1](x,pyramid[i])
        return x,prev






class RPDNet(nn.ModuleList):
    def __init__(self,in_channels=3,out_channels=3,features=32,iterations=6):
        super(RPDNet,self).__init__()

        self.iteratrions=iterations
        self.features=features

        self.body=RPDBlock(in_channels*2,ch=32)
        self.body_end=nn.Sequential(*[
            ResBlock(features,3,1,1),
            ResBlock(features,3,1,1),
            nn.Conv2d(features,out_channels,3,1,1)])


    def forward(self, Input):
        x=Input
        x_list=[]
        n,c,h,w=Input.size()

        prev = [[torch.zeros(n, self.features, h//4, w//4).to(Input.device),
                 torch.zeros(n, self.features, h//4, w//4).to(Input.device)],
                [torch.zeros(n, self.features, h//2, w//2).to(Input.device),
                 torch.zeros(n, self.features, h//2, w//2).to(Input.device)],
                [torch.zeros(n, self.features, h, w).to(Input.device),
                 torch.zeros(n, self.features, h, w).to(Input.device)]]

        for _ in range(self.iteratrions):
            x=torch.cat((Input,x),1)

            x,prev=self.body(x,prev)
            x=self.body_end(x)

            x=Input-x
            x_list.append(x)
        return x_list
