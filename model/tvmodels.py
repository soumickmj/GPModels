import torch
import torch.nn as nn
import torchvision.models

class TVModelWrapper(nn.Module):
    def __init__(self, model=torchvision.models.resnet18, in_channels=1, num_classes=1):
        super(TVModelWrapper, self).__init__()
        self.net = model(num_classes=num_classes)
        self.net.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.net.conv1.out_channels,
                                    kernel_size=self.net.conv1.kernel_size, stride=self.net.conv1.stride, padding=self.net.conv1.padding,
                                    dilation=self.net.conv1.dilation, groups=self.net.conv1.groups, bias=self.net.conv1.bias, padding_mode=self.net.conv1.padding_mode)

    def forward(self, x):
        return self.net(x)