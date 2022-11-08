##Unet Model##

import torch
import torch.nn as nn
import torch.nn.functional as F

#extra for PM (Pyramid Pooling Module)
import numpy as np
import torchvision.models as models
import math




def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
    )

# Single conv like florains
def single_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3), nn.ReLU(inplace=True)
    )    

#Crop func of down conv out before concatination to match TranseConve out size
def crop_img(tensor, target_tensor):         #(tensor: out of downconv, target_tensor: out of TransConve (smaller))
    target_size = target_tensor.size()[2]    #squared hxw
    tensor_size = tensor.size()[2]      #squared hxw
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[: ,:, delta:tensor_size - delta, delta:tensor_size - delta]     #return all batch sizes, all channels, height, width



class PM_UNet(nn.Module):
    def __init__(self, in_channels, stride):
    #def __init__(self):
        super(PM_UNet, self).__init__()

        ##########The Pyramid Pooling module part/ normal conv
        #  ref U-Net with spatial pyramid pooling for drusen segmentation in optical coherence tomography.pdf
        # code inspiered by: https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/models/psp.py

        #out_channels = int( in_channels / 4 )  #cause we have five elements in the pyramid,4 in pspnet for example
        out_channels = int( in_channels / 2 )   #output channel for the 4 elements = 8 = final desierd output 32/4 or inputchannel/2 following the Unet ssp.pdf Fig.1 


        out_channels = int( in_channels / 2 )
        out_channels2 = in_channels


        # max pool x to match the others when concatinating later in fwd
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=stride)

        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels2, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        
        ######not sure if we want those to cont like in:+DeepLab3 without dropout
        #### not sure if to include it cause it is outside the pyramid pooling module box with deopout
        #to get half of concatinate output Fig(e) = out*4 = in*2 = half of concat output
        self.fusion_bottleneck = nn.Sequential(nn.Conv2d(in_channels * 4, out_channels* 4, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(out_channels * 4),
                                               nn.ReLU(True),
                                               nn.Dropout2d(0.1, False))
        ###$$$could consider letting the outputchannel be = out_channel = 8 quarter the size for next layer in encoder and let double_conv(8,32) but doesn't match Fig1 in the Unet spatial.pdf



    def forward(self, x, stride):   #x is the feature_map
        # feature map shape (batch_size, in_channels, height/output_stride, width/output_stride)

        x_h = x.size()[2] # (== h/16)
        x_w = x.size()[3] # (== w/16)

        # print("x_h is" ,x_h)
        # print("x_w is" ,x_w)

        fcn_features_spatial_dim = x.size()[2:]
        # print("fcn_features_spatial_dim is" ,fcn_features_spatial_dim)

        print("x our ASPP input img size is" ,x.size())

        conv = self.conv1(x)
        print("conv 1x1 size" ,conv.size())

        atrous_1 = self.conv2(x)
        print("atrous_1 3x3 d 6 size" ,atrous_1.size())

        atrous_2 = self.conv3(x)
        print("atrous_2 3x3 d 12 size" ,atrous_2.size())

        atrous_3 = self.conv4(x)
        print("atrous_3 3x3 d 18 size" ,atrous_3.size())

        pooled_1 = self.avg_pool(x) 
        print("pool_1 2x2 size after avg_pool" ,pooled_1.size())
        pooled_1 = self.conv5(pooled_1)
        print("pool_1 2x2 size b4 UpSamp" ,pooled_1.size())
        #upsampling needed because of avg pooling 
        ######upsample to half h,w size for U-Net with spatial pyramid pooling for drusen segmentation in optical coherence tomography.pdf
        pooled_1 = nn.functional.upsample_bilinear(pooled_1, size=(int(x_h/stride), int(x_w/stride)))
        print("pool_1 2x2 size after UpSamp" ,pooled_1.size())


        # max pool x only when stride = 2 encoder part, to match the others when concatinating below
        if stride == 2:
            x = self.max_pool_2x2(x)

        # with x like in the pyramiid pooling Module
        # dim=1 means that tensor size of each element should be the same except for dim=1, which is no of channels all have 6 but x has 32 = 6x5 elements
        x = torch.cat([x, conv, atrous_1, atrous_2, atrous_3, pooled_1],
                        dim=1)
        # print("cat ans", x.size())

        #####not sure
        # witout x like in +DeepLab3
        # x = torch.cat([pooled_1, pooled_2, pooled_3, pooled_4, pooled_5],
        #                dim=1)

        #### not sure if to include it cause it is outside the pyramid pooling module box
        # Concatination output is twice the desired output so we use con1x1 to get half of it 
        x = self.fusion_bottleneck(x)
        # print("x size is" ,x.size())

        return x






class UNet(nn.Module):
    def __init__(self, in_channels):
    #def __init__(self):
        super(UNet, self).__init__()

        
        ##########The Pyramid Pooling module part/ normal conv, still need to be replaced by dialated conv
        #  ref U-Net with spatial pyramid pooling for drusen segmentation in optical coherence tomography.pdf
        # code inspiered by: https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/models/psp.py

        out_channels = int( in_channels / 4 )

        ###self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        # self.conv5 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
        #                            nn.BatchNorm2d(out_channels),
        #                            nn.ReLU(True))
        
        
        #not sure if to include it cause it is outside the pyramid pooling module box
        self.fusion_bottleneck = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels *4, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(out_channels *4),
                                               nn.ReLU(True),
                                               nn.Dropout2d(0.1, False))
        
        
        # ##############Unet part
        # #Down convolution part
        # # Shallower model like florians
        # self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.down_conv_1 = double_conv(1, 32) 
        # self.down_conv_2 = double_conv(32, 64) 
        # self.down_conv_3 = single_conv(64, 128)     #not double in florians

        # # Original Unet
        # # self.down_conv_1 = double_conv(1, 64) 
        # # self.down_conv_2 = double_conv(64, 128) 
        # # self.down_conv_3 = double_conv(128, 256) 
        # # self.down_conv_4 = double_conv(256, 512) 
        # # self.down_conv_5 = double_conv(512, 1024) 

        # #Trans Conv with channel cropping for concatinating with previous down conv
        # self.up_trans_1 = nn.ConvTranspose2d(
        #     in_channels=128,                             #original unit 1024,
        #     out_channels=64,                             #original unit 512,
        #     kernel_size=2,
        #     stride=2)

        # #we crop the channel 128 -> 64 cause will cocatinate with encoding output at same level = 64,
        # #therefore next conv in channels = 64+64 = 126

        # self.up_conv_1 = single_conv(128, 64)      #in_channels , out_Channels  original unit (1024, 512)
        # #self.up_conv_1 = double_conv(1024, 512)   #in_channels , out_Channels  original unit (1024, 512)




        # self.up_trans_2 = nn.ConvTranspose2d(
        #     in_channels=64,                               # original unit 512,
        #     out_channels=32,                              # original unit 256,
        #     kernel_size=2,
        #     stride=2)

        # self.up_conv_2 = single_conv(64, 32)   #in_channels , out_Channels  original unit (512, 256)
        # #self.up_conv_2 = double_conv(512, 256)   #in_channels , out_Channels  original unit (512, 256)


        # ## Cont original unit
        # # self.up_trans_3 = nn.ConvTranspose2d(
        # #     in_channels=256,
        # #     out_channels=128,
        # #     kernel_size=2,
        # #     stride=2)

        # # self.up_conv_3 = double_conv(256, 128)   #in_channels , out_Channels        




        # # self.up_trans_4 = nn.ConvTranspose2d(
        # #     in_channels=128,
        # #     out_channels=64,
        # #     kernel_size=2,
        # #     stride=2)

        # # self.up_conv_4 = double_conv(128, 64)   #in_channels , out_Channels    


        # ### For Classification and following Florian
        # #Last 2d Convolutional layer with kernel size =1
        # self.out = nn.Sequential(
        #     nn.Conv2d(32, 4, kernel_size=1),    #original unit 64, 1. However, following Florians =32 and used as class not seg. out is = no of labels here =4 cause of hot encoding
        #     nn.Softmax(dim=1)
        # )
      
        # # #regression
        # # self.out = nn.Sequential(
        # #     nn.Conv2d(32, 4, kernel_size=1),    #original unit 64, 1. However, following Florians =32 and used as class not seg. out is = no of labels here =4 cause of hot encoding
        # #     nn.Tanh()
        # # )


        # # ## Original Unet
        # # #Last 2d Convolutional layer with kernel size =1  
        # # self.out = nn.Conv2d(
        # #     in_channels=64, 
        # #     out_channels=1,       #in the original Seg paper is 2, for us is one cause we have tumor 1 and background, but if there is multiple obj to segment, increase output channels
        # #     kernel_size=1
        # # )

  


        # #Last layer:  fully connected Conv layer for classification purposes
        # # self.fc = nn.Sequential(nn.Linear(64*324*324, 128), #in features = (n_features_conv * height * width)   n_features_conv: is the no of channels out of conv
        # #                         nn.SELU(),
        # #                         nn.Dropout(p=0.4),
        # #                         #  nn.Linear(h1, h2),      
        # #                         #  nn.SELU(),
        # #                         #  nn.Dropout(p=0.4),
        # #                         nn.Linear(128, 3),
        # #                         nn.LogSigmoid())

        # # self.fc_linear1 = nn.Linear(1024*12*12, 1024)         #1024*12*12 / 144 = 1024
        # # self.fc_selu = nn.SELU()
        # # self.fc_drop = nn.Dropout(p=0.4)
        # # self.fc_linear2 = nn.Linear(1024, 4)
        # # self.fc_sig = nn.LogSigmoid()



        # #advanced used to initialize the weights manually https://discuss.pytorch.org/t/pytorch-self-module/49677/3
        # # for m in self.modules():
            
        # #     if isinstance(m, nn.Conv2d):
        # #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # #         m.weight.data.normal_(0, math.sqrt(2. / n))
                
        # #     elif isinstance(m, nn.BatchNorm2d):
        # #         m.weight.data.fill_(1)
        # #         m.bias.data.zero_()


        
    def forward(self, x):

        fcn_features_spatial_dim = x.size()[2:]

        print("x size is" ,x.size())

        pooled_1 = nn.functional.adaptive_avg_pool2d(x, 1)
        pooled_1 = self.conv1(pooled_1)
        pooled_1 = nn.functional.upsample_bilinear(pooled_1, size=fcn_features_spatial_dim)
        print("pool_1 size b4 UpSamp" ,pooled_1.size())

        pooled_2 = nn.functional.adaptive_avg_pool2d(x, 2)
        pooled_2 = self.conv2(pooled_2)
        pooled_2 = nn.functional.upsample_bilinear(pooled_2, size=fcn_features_spatial_dim)
        print("pool_2 size b4 UpSamp" ,pooled_2.size())

        pooled_3 = nn.functional.adaptive_avg_pool2d(x, 3)
        pooled_3 = self.conv3(pooled_3)
        pooled_3 = nn.functional.upsample_bilinear(pooled_3, size=fcn_features_spatial_dim)
        print("pool_3 size b4 UpSamp" ,pooled_3.size())

        pooled_4 = nn.functional.adaptive_avg_pool2d(x, 6)
        pooled_4 = self.conv4(pooled_4)
        pooled_4 = nn.functional.upsample_bilinear(pooled_4, size=fcn_features_spatial_dim)
        print("pool_4 size b4 UpSamp" ,pooled_4.size())

        # pooled_5 = nn.functional.adaptive_avg_pool2d(x, 16)
        # pooled_5 = self.conv5(pooled_5)
        # pooled_5 = nn.functional.upsample_bilinear(pooled_5, size=fcn_features_spatial_dim)

        x = torch.cat([x, pooled_1, pooled_2, pooled_3, pooled_4],            #, pooled_5
                       dim=1)


        #not sure if to include it cause it is outside the pyramid pooling module box
        x = self.fusion_bottleneck(x)
        print("x size is" ,x.size())

        return x

        

        

        

    # def forward(self, image):
    #     # batch size, Channel, height, width
    #     # encoder part

    #     #print(image.size())

    #     image = F.pad(image, (2, 2, 2, 2))  # [left, right, top, bot]  # Double conv self padding alternative for pytorch inpit size = output size
    #     x1 = self.down_conv_1(image) #     #first (double conv + ReLU) to input
    #     print(x1.size())
    #     x2 = self.max_pool_2x2(x1)        #Followed by 2x2 Max pooling
    #     x2 = F.pad(x2, (2, 2, 2, 2))      #Double conv self padding alternative for pytorch inpit size = output size
    #     x3 = self.down_conv_2(x2)    #    #repeated 5 times withoyt a 5th max pool
    #     print(x3.size())
    #     x4 = self.max_pool_2x2(x3)   
    #     x4 = F.pad(x4, (1, 1, 1, 1))      #Single conv self padding alternative for pytorch inpit size = output size
    #     x5 = self.down_conv_3(x4)    
    #     print(x5.size())                #should be (8, 128 ,127, 127)  (batchxAug, channel, h, w)

    #     # #cont fo original Unet
    #     # x6 = self.max_pool_2x2(x5)      
    #     # x7 = self.down_conv_4(x6)    # 
    #     # x8 = self.max_pool_2x2(x7)        
    #     # x9 = self.down_conv_5(x8)    
    #     # #print(x9.size())

    #     # #max pooling for ol classification  attempts of original unet output purpose not Unet Seg
    #     # x10 = self.max_pool_2x2(x9)    
    #     # #print(x10.size())

    #     #decoder part

    #     #UpTrans 1
    #     x = self.up_trans_1(x5)   #last output value  # Originally x9
    #     print(x.size())

    #     # no crop needed same padding inputsize=outputsize
    #     #y = crop_img(x7, x)      #Crop func of down conv out before concatination to match TranseConve out size # Originally x7
    #     #x = self.up_conv_1(torch.cat([x, y], 1))     #cobcatenate x and cropped x7=y, axis=1
        
    #     # instead concatinate the images first and then pad prior to single conv
    #     x = torch.cat([x, x3], 1)
    #     print(x.size())
    #     x = F.pad(x, (1, 1, 1, 1))      #Single conv self padding alternative for pytorch inpit size = output size
    #     x = self.up_conv_1(x)     #cobcatenate x and cropped x7=y, axis=1, Originally x9
    #     print(x.size())

    #     #UpTrans 2
    #     x = self.up_trans_2(x)   #last output value
    #     print(x.size())

    #     # no crop needed same padding inputsize=outputsize
    #     #y = crop_img(x5, x)      #Crop func of down conv out before concatination to match TranseConve out size
    #     #x = self.up_conv_2(torch.cat([x, y], 1))     #cobcatenate x and cropped x7=y, axis=1, Originally x9
        
    #     # instead concatinate the images first and then pad prior to single conv
    #     x = torch.cat([x, x1], 1)
    #     print(x.size())
    #     x = F.pad(x, (1, 1, 1, 1))      #Single conv self padding alternative for pytorch inpit size = output size
    #     x = self.up_conv_2(x)     #cobcatenate x and cropped x7=y, axis=1, Originally x9
    #     print(x.size())        



    #     ##### cont original unet
    #     # #UpTrans 3
    #     # x = self.up_trans_3(x)   #last output value
    #     # y = crop_img(x3, x)      #Crop func of down conv out before concatination to match TranseConve out size 
    #     # x = self.up_conv_3(torch.cat([x, y], 1))     #cobcatenate x and cropped x7=y, axis=1
    #     # #print(x.size())
 

    #     # #UpTrans 4
    #     # x = self.up_trans_4(x)  #last output value
    #     # y = crop_img(x1, x)      #Crop func of down conv out before concatination to match TranseConve out size 
    #     # x = self.up_conv_4(torch.cat([x, y], 1))     #cobcatenate x and cropped x7=y, axis=1
    #     # #print(x.size())
        
    #     # ## Segmentation
    #     # #Add the output channel
    #     # Seg = self.out(x)
    #     # #print(x.size())




    #     ## Right Classification output
    #     #add last 2d sigle kernel size= 1 conv with softmax activation func as classification output
    #     ### self padding not needed for 2d conv with kernel size = 1, size doesn't change
    #     Class = self.out(x)
    #     print(Class.size())





    #     # ## Old Clasification attempts
    #     # # Flatten x with start_dim=1 for fully connected layer for encoder part/classification
    #     # x = torch.flatten(x10, 1)
    #     # #print(x.size())

    #     # #Add the output channel from fully connected layer
    #     # #x = self.fc(x)
    #     # Class = self.fc_sig(self.fc_linear2(self.fc_drop(self.fc_selu(self.fc_linear1(x)))))

    #     #print("final segmentation Output")
    #     #print(Seg.size())
    #     #print("final classification Output")
    #     #print(Class.size())
        
        
    #     return Class 
    #     #return Seg, Class 
    #     # return Seg 
    #     #return Class 
        



#to run it here from this script, uncomment the following

if __name__ == "__main__":                #to run it
    image = torch.rand(8, 32, 508, 508)    #specify your image: batch size, Channel, height, width
    model = UNet(in_channels = 32)                        #Initialize the model
    print(model(image))
