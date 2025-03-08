# Adapted from https://github.com/soumickmj/FTSuperResDynMRI/blob/main/models/unet2d.py

import torch
from torch import nn
import torch.nn.functional as F
import torchcomplex.nn.functional as cF



__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"


class GP_UNet(nn.Module):
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
                 batch_norm=False, up_mode='upconv', dropout=False, Relu = "Relu", out_act="None"): #dropout=False
        super(GP_UNet, self).__init__()
        assert up_mode in ('upconv', 'bilinear', 'sinc', "upsample_Sinc")
        assert out_act in ("softmax", "None", "sigmoid", "relu")
        self.padding = padding
        self.depth = depth
        self.Relu = Relu
        self.dropout = nn.Dropout2d() if dropout else nn.Sequential()
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm, Relu))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm, Relu))
            prev_channels = 2**(wf+i)

        if out_act == "softmax":
            self.last = nn.Sequential(
                      nn.Conv2d(prev_channels, n_classes, kernel_size=1),
                      nn.Softmax2d()
                    )
        
        elif out_act == "sigmoid":
            self.last = nn.Sequential(
                      nn.Conv2d(prev_channels, n_classes, kernel_size=1),
                      nn.Sigmoid()
                    )
        
        elif out_act == "relu":
            self.last = nn.Sequential(
                      nn.Conv2d(prev_channels, n_classes, kernel_size=1),
                      nn.ReLU()
                    )

        else:
            self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        

        ### For Classification, following Florian's GP-UNet
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                #x = nn.AvgPool2d(x, 2)
                x = F.avg_pool2d(x, 2)
        x = self.dropout(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        if self.training:
            x = self.GMP(x)
            return self.last(x).view(x.shape[0],-1)
        else:
            mask = self.last(x)
            x = self.GMP(x)
            pred = self.last(x).view(x.shape[0],-1)
            return pred, mask

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, Relu):
        super(UNetConvBlock, self).__init__()
        block = [nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding))]

        if Relu == "Relu":
            block.append(nn.ReLU())
        else:
            block.append(nn.PReLU())

        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))

        if Relu == "Relu":
            block.append(nn.ReLU())
        else:
            block.append(nn.PReLU())

        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, Relu):
        super(UNetUpBlock, self).__init__()

        self.up_mode = up_mode

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'bilinear':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),      #'trilinear'
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        elif 'inc' in up_mode:   
            self.up = nn.Conv2d(in_size, out_size, kernel_size=1)


        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, Relu)

    def forward(self, x, bridge):
        if self.up_mode in ['upconv', 'bilinear']: #  'upconv'
            up = self.up(x)
        elif 'inc' in self.up_mode: 
            x = cF._sinc_interpolate(x, size=[int(x.shape[2]*2), int(x.shape[3]*2)]) #'sinc' ###sth wrong
            up = self.up(x)

        # bridge = self.center_crop(bridge, up.shape[2:]) #sending shape ignoring 2 digit, so target size start with 0,1,2
        up = F.interpolate(up, size=bridge.shape[2:], mode='bilinear')
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out

#to run it here from this script, uncomment the following

if __name__ == "__main__":                #to run it
    image = torch.rand(2, 4, 240, 240)    #specify your image: batch size, Channel, height, width
    model = GP_UNet(in_channels=4, n_classes=3, depth=4, wf=6, up_mode="upsample_Sinc", Relu = "Relu")          #Initialize the model, up_mode = "upconv" or "upsample1" == interpolate mode Bilinear or "upsample" == interpolate mode sinc
    model.eval()
    out = model(image)
    print(model(image))
