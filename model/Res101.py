import torch
import torch.nn as nn
import torchvision.models

import torch.nn.functional as F

class ResNet101(nn.Module):
    def __init__(self, model=torchvision.models.resnet101, in_channels=1, num_classes=1):
        super(ResNet101, self).__init__()
        self.net = model(num_classes=num_classes)
        self.net.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.net.conv1.out_channels,
                                    kernel_size=self.net.conv1.kernel_size, stride=self.net.conv1.stride, padding=self.net.conv1.padding,
                                    dilation=self.net.conv1.dilation, groups=self.net.conv1.groups, bias=self.net.conv1.bias, padding_mode=self.net.conv1.padding_mode)
        
    def forward(self, x):
        return self.net(x)



#to run it here from this script, uncomment the following
import torchvision.models as models
if __name__ == "__main__":                #to run it
    image = torch.rand(2, 4, 240, 240)    #specify your image: batch size, Channel, height, width
    model = ResNet101(model=models.resnet101, in_channels=4, num_classes=3)
    model.eval()
    out = model(image)
    print(model(image))
