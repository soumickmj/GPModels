# Adapted from https://github.com/soumickmj/FTSuperResDynMRI/blob/main/models/unet2d.py

import torch
from torch import nn
import torch.nn.functional as F
import torchcomplex.nn.functional as cF
#PM part
import math

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

##Modified in 25th June 2021 to include spatial pyramid pooling module inspired by https://arxiv.org/abs/1912.05404.
__author__="Hadya Yassin"
__copyright__="Copyright 2021 Hadya Yassin"
__credits__ = ["Hadya Yassin"]
__license__ = "Apache"
__version__ = "2.0"
__maintainer__ = "Hadya Yassin"
__email__ = "hadya.yassin@ovgu.de"
__status__ = "Production"


class GP_UNet_PM(nn.Module):
    """
    Implementation of
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (Ronneberger et al., 2015)
    https://arxiv.org/abs/1505.04597

    Using the default arguments will yield the exact version used
    in the original paper

    Args:
        in_channels (int): number of input channels
        n_classes (int): number of output channels
        depth (int): depth of the network
        wf (int): number of filters in the first layer is 2**wf
        padding (bool): if True, apply padding such that the input shape
                        is the same as the output.
                        This may introduce artifacts
        batch_norm (bool): Use BatchNorm after layers with an
                            activation function
        up_mode (str): one of 'upconv' or 'upsample'.
                        'upconv' will use transposed convolutions for
                        learned upsampling.
                        'upsample_Bi' will use bilinear upsampling.
                        'upsample_Sinc' will use sinc upsampling.
    """
    def __init__(self, in_channels=1, n_classes=1, depth=3, wf=6, padding=True,
                 batch_norm=False, up_mode='upconv', dropout=False, out_act="softmax"):
        super(GP_UNet_PM, self).__init__()
        assert up_mode in ('upconv', 'bilinear', 'sinc', "upsample_Sinc")
        self.padding = padding
        self.depth = depth
        self.dropout = nn.Dropout2d() if dropout else nn.Sequential()
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.downS = nn.ModuleList()
        self.downS2 = nn.ModuleList()
        for i in range(depth):
            if in_channels==1 or in_channels==4:
                self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
                in_channels = 0
                self.downS.append(SingleConvBlock(2**(wf+i), int(2**(wf+i)/4), batch_norm=False))
                self.downS2.append(SingleConvBlock(2**(wf+i), int(2**(wf+i)/2), batch_norm=False))
                
            else:
                prev_channels = 2**(wf+i)
                self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
                self.downS.append(SingleConvBlock(2**(wf+i), int(2**(wf+i)/4), batch_norm=False))
                self.downS2.append(SingleConvBlock(2**(wf+i), int(2**(wf+i)/2), batch_norm=False))
                

        self.up_path = nn.ModuleList()
        self.upS2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)
            self.upS2.append(SingleConvBlock(2**(wf+i), int(2**(wf+i)/2), batch_norm=False))

        if out_act == "softmax":
            self.last = nn.Sequential(
                      nn.Conv2d(prev_channels, n_classes, kernel_size=1),
                      nn.Softmax2d()
                    )
        else:
            self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        ### For Classification, following Florian's GP-UNet
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))


    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)     #downconv
            if i != len(self.down_path)-1:
                blocks.append(x)        #which is the #bridge idea is the x from left side at each loop downward conv
                pm = PM(self.downS[i], self.downS2[i], x, PM_mode="down", levels=[1, 2, 3, 6, 16])    #pm intead of avg pooling
                x = F.max_pool2d(x, 2)
                # x = F.avg_pool2d(x, 2)
                x = self.downS2[i](x)
                x = torch.cat([x, pm[0], pm[1], pm[2], pm[3], pm[4]], 1)  #bridge is the x from left side at each loop down conv


        x = self.dropout(x)

        c=3     #counter

        for i, up in enumerate(self.up_path):
            x = up(self.downS[c], self.downS2[c], self.upS2[i], x, blocks[-i-1])
            c = c - 1

        if self.training:
            x = self.GMP(x)
            return self.last(x).view(x.shape[0],-1)
        else:
            mask = self.last(x)
            x = self.GMP(x)
            pred = self.last(x).view(x.shape[0],-1)
            return pred, mask

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()

        self.up_mode = up_mode

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'bilinear':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),      #'trilinear'
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        elif 'inc' in  up_mode:   
            self.up = nn.Conv2d(in_size, out_size, kernel_size=1)

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def forward(self, downS, downS2, upS2, x, bridge):
        if self.up_mode == 'upconv': #  'upconv'
            up = self.up(x)
        elif self.up_mode == 'bilinear':
            up = self.up(x)
        elif 'inc' in  self.up_mode:
            x = cF._sinc_interpolate(x, size=[int(x.shape[2]*2), int(x.shape[3]*2)]) #'sinc' ###sth wrong
            up = self.up(x)

        # bridge = self.center_crop(bridge, up.shape[2:]) #sending shape ignoring 2 digit, so target size start with 0,1,2
        up = F.interpolate(up, size=bridge.shape[2:], mode='bilinear')
        up = upS2(up) 
        bridge = PM(downS, downS2, bridge, PM_mode="bridge", levels=[1, 2, 3, 6, 16])
        out = torch.cat([up, bridge[0],bridge[1],bridge[2],bridge[3],bridge[4]], 1)  #bridge is the x from left side at each loop down conv
        out = self.conv_block(out)
        # print(out.shape)

        return out


class SingleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm):
        super(SingleConvBlock, self).__init__()
        blockS = []

        blockS.append(nn.Conv2d(in_size, out_size, kernel_size=1))
        blockS.append(nn.ReLU())
        if batch_norm:
            blockS.append(nn.BatchNorm2d(out_size))

        self.blockS = nn.Sequential(*blockS)

    def forward(self, x):
        outS = self.blockS(x)
        return outS



def PM(downS, downS2, x, PM_mode, levels):

    pool = []

    #for i in range(0, 5): #loop 5 times for 5 levels
    for i in range(len(levels)): #loop 5 times for 5 levels
        p = F.avg_pool2d(x, math.ceil(x.shape[2]/levels[i]), math.floor(x.shape[2]/levels[i]))  #resulting pool bin 1x1, 2x2, 3x3, 6x6, 16x16
        pool.append(p) 


    Singleconv = []

    for i in range(len(levels)): #loop 5 times for 5 levels
        #1x1 conv to reduce num of channels N/2 for 2x2 bin and N/4 for the rest 4
        if levels[i] == 2:
            S = downS2(pool[i])
        else:
            S = downS(pool[i])

        #Upsampling"
        if PM_mode == "down":
            S = F.interpolate(S, size=(int(x.shape[2]/2), int(x.shape[3]/2)), mode='bilinear')
        elif PM_mode == "bridge":
            S = F.interpolate(S, size=x.shape[2:], mode='bilinear')

        Singleconv.append(S)


    outSPP = Singleconv


    return outSPP


#to run it here from this script, uncomment the following

if __name__ == "__main__":                #to run it
    image = torch.rand(2, 4, 240, 240)    #specify your image: batch size, Channel, height, width
    model = GP_UNet_PM(in_channels=4, n_classes=3, depth=5, wf=4, up_mode="sinc")       #Initialize the model, up_mode = "upconv" or "upsample_Bi" == interpolate mode Bilinear or "upsample_Sinc" == interpolate mode sinc
    model.eval()
    out = model(image)
    # print(model(image))
