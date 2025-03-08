# Adapted from https://raw.githubusercontent.com/soumickmj/NCC1701/main/Bridge/models/ResNet/MickResNet.py

import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
from tricorder.torch.transforms import Interpolator

__author__ = "Soumick Chatterjee"
__copyright__ = "Copyright 2019, Soumick Chatterjee & OvGU:ESF:MEMoRIAL"
__credits__ = ["Soumick Chatterjee"]

__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Published"

class ResidualBlock(nn.Module):
    def __init__(self, in_features, drop_prob=0.2): #drop_prob=0.2
        super(ResidualBlock, self).__init__()

        conv_block = [  layer_pad(1),
                        layer_conv(in_features, in_features, 3),
                        layer_norm(in_features),
                        act_relu(),
                        layer_drop(p=drop_prob, inplace=True),
                        layer_pad(1),
                        layer_conv(in_features, in_features, 3) ,
                        layer_norm(in_features) ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class DownsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownsamplingBlock, self).__init__()

        conv_block = [  layer_conv(in_features, out_features, 3, stride=2, padding=1),
                        layer_norm(out_features),
                        act_relu()  ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)

class UpsamplingBlock(nn.Module):
    def __init__(self, in_features, out_features, mode="upconv", interpolator=None, post_interp_convtrans=False):
        super(UpsamplingBlock, self).__init__()

        self.interpolator = interpolator
        self.mode = mode
        self.post_interp_convtrans = post_interp_convtrans
        if self.post_interp_convtrans:
            self.post_conv = layer_conv(out_features, out_features, 1)

        if mode == "upconv":
            conv_block = [  layer_convtrans(in_features, out_features, 3, stride=2, padding=1, output_padding=1),   ]
        else:
            conv_block = [  layer_pad(1),
                            layer_conv(in_features, out_features, 3),   ]
        conv_block += [ layer_norm(out_features),
                        act_relu()  ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x, out_shape=None):
        if self.mode != "upconv":
            return self.conv_block(self.interpolator(x, out_shape))
        if not self.post_interp_convtrans:
            return self.conv_block(x)
        x = self.conv_block(x)
        return (
            self.post_conv(self.interpolator(x, out_shape))
            if x.shape[2:] != out_shape
            else x
        )

class GP_ReconResNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, res_blocks=14, starting_nfeatures=64, updown_blocks=2, is_relu_leaky=True, do_batchnorm=False, res_drop_prob=0.2,    #res_drop_prob=0.2
                        out_act="softmax", forwardV=0, upinterp_algo='upconv', post_interp_convtrans=False, is3D=False): #should use 14 as that gives number of trainable parameters close to number of possible pixel values in a image 256x256 
        super(GP_ReconResNet, self).__init__()

        layers = {}
        if is3D:
            sys.exit("ResNet: for implemented for 3D, ReflectionPad3d code is required")
            layers["layer_conv"] = nn.Conv3d
            layers["layer_convtrans"] = nn.ConvTranspose3d
            layers["layer_norm"] = nn.BatchNorm3d if do_batchnorm else nn.InstanceNorm3d
            layers["layer_drop"] = nn.Dropout3d
            layers["layer_pad"] = ReflectionPad3d
            layers["interp_mode"] = 'trilinear'
        else:
            layers["layer_conv"] = nn.Conv2d
            layers["layer_convtrans"] = nn.ConvTranspose2d
            layers["layer_norm"] = nn.BatchNorm2d if do_batchnorm else nn.InstanceNorm2d
            layers["layer_drop"] = nn.Dropout2d
            layers["layer_pad"] = nn.ReflectionPad2d
            layers["interp_mode"] = 'bilinear'
        layers["act_relu"] = nn.PReLU if is_relu_leaky else nn.ReLU
        globals().update(layers)

        self.forwardV = forwardV
        self.upinterp_algo = upinterp_algo

        interpolator = Interpolator(mode=layers["interp_mode"] if self.upinterp_algo == "upconv" else self.upinterp_algo)

        in_channels = in_channels
        out_channels = n_classes
        # Initial convolution block
        intialConv = [  layer_pad(3),
                        layer_conv(in_channels, starting_nfeatures, 7),
                        layer_norm(starting_nfeatures),
                        act_relu() ]

        # Downsampling [need to save the shape for upsample]
        downsam = []
        in_features = starting_nfeatures
        out_features = in_features*2
        for _ in range(updown_blocks):
            downsam.append(DownsamplingBlock(in_features, out_features))
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        resblocks = [
            ResidualBlock(in_features, res_drop_prob) for _ in range(res_blocks)
        ]

        # Upsampling
        upsam = []
        out_features = in_features//2
        for _ in range(updown_blocks):
            upsam.append(UpsamplingBlock(in_features, out_features, self.upinterp_algo, interpolator, post_interp_convtrans))
            in_features = out_features
            out_features = in_features//2

        # Output layer
        finalconv = [   layer_conv(starting_nfeatures, out_channels, 1),    ] #kernel size changed from 7 to 1 to make GMP work

        if out_act == "sigmoid":
            finalconv += [   nn.Sigmoid(), ]
        elif out_act == "relu":
            finalconv += [   act_relu(), ]
        elif out_act == "tanh":
            finalconv += [   nn.Tanh(), ]
        elif out_act == "softmax":
            finalconv += [   nn.Softmax2d(), ]


        self.intialConv = nn.Sequential(*intialConv)
        self.downsam = nn.ModuleList(downsam)
        self.resblocks = nn.Sequential(*resblocks)
        self.upsam = nn.ModuleList(upsam)
        self.finalconv = nn.Sequential(*finalconv)

        ### For Classification, following Florian's GP-UNet
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))

        if self.forwardV == 0:
            self.forward = self.forwardV0
        elif self.forwardV == 1:
            sys.exit("ResNet: its identical to V0 in case of GP_ResNet")
        elif self.forwardV == 2:
            self.forward = self.forwardV2
        elif self.forwardV == 3:
            self.forward = self.forwardV3
        elif self.forwardV == 4:
            self.forward = self.forwardV4
        elif self.forwardV == 5:
            self.forward = self.forwardV5

    def final_step(self, x):    
        if self.training:
            x = self.GMP(x)
            return self.finalconv(x).view(x.shape[0],-1)
        else:
            mask = self.finalconv(x)
            x = self.GMP(x)
            pred = self.finalconv(x).view(x.shape[0],-1)
            return pred, mask
            
    def forwardV0(self, x):
        #v0: Original Version
        x = self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(x.shape[2:])
            x = downblock(x)
        x = self.resblocks(x)
        for i, upblock in enumerate(self.upsam):
            x = upblock(x, shapes[-1-i])
        return self.final_step(x)

    def forwardV2(self, x):
        #v2: residual of v1 + input to the residual blocks added back with the output
        out = self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(out.shape[2:])
            out = downblock(out)
        out = out + self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        return self.final_step(out)

    def forwardV3(self, x):
        #v3: residual of v2 + input of the initial conv added back with the output
        out = x + self.intialConv(x)
        shapes = []
        for downblock in self.downsam:
            shapes.append(out.shape[2:])
            out = downblock(out)
        out = out + self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        return self.final_step(out)

    def forwardV4(self, x):
        #v4: residual of v3 + output of the initial conv added back with the input of final conv
        iniconv = x + self.intialConv(x)
        shapes = []
        if len(self.downsam) > 0:
            for i, downblock in enumerate(self.downsam):
                if i == 0:
                    shapes.append(iniconv.shape[2:])
                    out = downblock(iniconv)
                else:
                    shapes.append(out.shape[2:])
                    out = downblock(out)
        else:
            out = iniconv
        out = out + self.resblocks(out)
        for i, upblock in enumerate(self.upsam):
            out = upblock(out, shapes[-1-i])
        out = iniconv + out
        return self.final_step(out)

    def forwardV5(self, x):
        #v5: residual of v4 + individual down blocks with individual up blocks
        outs = [x + self.intialConv(x)]
        shapes = []
        for downblock in self.downsam:
            shapes.append(outs[-1].shape[2:])
            outs.append(downblock(outs[-1]))
        outs[-1] = outs[-1] + self.resblocks(outs[-1])
        for i, upblock in enumerate(self.upsam):
            outs[-1] = upblock(outs[-1], shapes[-1-i])
            outs[-1] = outs[-2] + outs.pop()
        return self.final_step(outs.pop())

#to run it here from this script, uncomment the following

if __name__ == "__main__":                #to run it
    image = torch.rand(2, 1, 240, 240)    #specify your image: batch size, Channel, height, width
    model = GP_ReconResNet(in_channels=1, n_classes=3, upinterp_algo='sinc')                        #Initialize the model
    # model.eval()
    out = model(image)
    print(model(image))
