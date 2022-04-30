# Adapted from https://github.com/soumickmj/FTSuperResDynMRI/blob/main/models/unet2d.py

import torch
from torch import nn
import torch.nn.functional as F




__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

##Modified in 23rd August 2021 to replace avgpooling dowinsapling with pixel unshuffle and replace upsampling (upcov and upsample interpolation) with pixel shuffle inspired by https://arxiv.org/pdf/2102.12898.pdf
__author__="Hadya Yassin"
__copyright__="Copyright 2021 Hadya Yassin"
__credits__ = ["Hadya Yassin"]
__license__ = "Apache"
__version__ = "2.0"
__maintainer__ = "Hadya Yassin"
__email__ = "hadya.yassin@ovgu.de"
__status__ = "Production"


class GP_UNet_miniShuffle(nn.Module):
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

        down_mode (str): 'pixelunshuffle'.
                    
        up_mode (str): 'pixelshuffle'.
                        'pixelshuffle' will use sub-pixel convolutions for
                        learned upsampling
                    
    """
    def __init__(self, in_channels=1, n_classes=1, depth=3, wf=6, padding=True,
                 batch_norm=False, down_mode='pixelunshuffle', up_mode='pixelshuffle', dropout=False, out_act="softmax"):
        super(GP_UNet_miniShuffle, self).__init__()
        assert down_mode in ('averagepooling', 'pixelunshuffle')
        assert up_mode in ('upconv', 'upsample', 'pixelshuffle')
        self.padding = padding
        self.depth = depth
        self.dropout = nn.Dropout2d() if dropout else nn.Sequential()
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.down_mode = down_mode

        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.pixel_unshuffle = nn.PixelUnshuffle(2)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

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
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                
                if self.down_mode == "averagepooling":
                    #x = nn.AvgPool2d(x, 2)
                    x = F.avg_pool2d(x, 2)

                elif self.down_mode == "pixelunshuffle":
                    # pixel_unshuffle = nn.PixelUnshuffle(2)
                    Pixel_Un = self.pixel_unshuffle(x)
                    Pixel_Un = Pixel_Un
                    conv1x1 = nn.Conv2d(int(Pixel_Un.shape[1]), int(x.shape[1]), kernel_size=1).cuda()
                    x = conv1x1(Pixel_Un)

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

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),      #'trilinear'
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        elif up_mode == 'pixelshuffle':
            self.up = nn.Sequential(nn.PixelShuffle(upscale_factor=2),      #'trilinear'
                                    nn.Conv2d(in_size//4, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def forward(self, x, bridge):
        up = self.up(x)        
        # bridge = self.center_crop(bridge, up.shape[2:]) #sending shape ignoring 2 digit, so target size start with 0,1,2
        up = F.interpolate(up, size=bridge.shape[2:], mode='bilinear')
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out

#to run it here from this script, uncomment the following

if __name__ == "__main__":                #to run it
    image = torch.rand(2, 1, 512, 512)    #specify your image: batch size, Channel, height, width
    model = GP_UNet_miniShuffle(in_channels=1, n_classes=3, depth=4, wf=6, down_mode='pixelunshuffle', up_mode='pixelshuffle')      #'pixelshuffle'  Initialize the model
    model.eval()
    out = model(image)
    print(model(image))
