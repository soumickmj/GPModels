from find_lr import find_lr


# Finding the optimum learning rate for the used dataset

"""

This is the main module for finding a suitable learning rate for a specific dataset:
The dataset consists of gray scale 512x512 images of specific orientation or all:
1- Sagittal
2- Coronal
3- Axial
4- All

Data augumented as folows:
1. Random Rotation between -330 and 330
2. Horizontal flip
3. Vertical flip
4. Randomly applied bspline deformation
5. Randomly applied colorJitter, randomly changing brightness, contrast and saturation of the image
6. Randomly applied motion artifact
7. Randomly Downscaling and upscaling, decreasing image quality


"""
"""
__author__ = "Hadya Yassin"
Created on Thu March 20 15:00:00 2021

"""

# useful libraries

import torch
#torch.manual_seed(17)
#torch.manual_seed(14041931)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models             #for resnet transfer learning model
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#Importing Dataset Agumentation class
from AugumentedDataset import AugumentedDataset

#Importing healthy brain check function
from healthy_brains import healthy_brains

#Importing the Models
from Resnet.GP_Resnet import GP_ResNet50  ####to import the unet model
# from Unet.GP_Unet import GP_UNet  ####to import the unet model

from prettytable import PrettyTable



###########################Specify changable parameters#######################################

#Main path used throuout the code
main_path = '/run/media/hyassin/New Volume/CentOS/MasterThesis2021/master_Yassin_Final'



### Image and batch parameters:
#the total num of Augumented images of each input image + the input image
#based on class AugumentedDataset below, default = 8
aug = 8

#Specify Image orientation
orient = 'All'         #All, Sag, Cor, Axi

#the desiered batch size
batch = 1              #for Training and validation
batch_test = 2         #for Testing



#Main path + specific path to pickle files, our input data
training_data_seg = pickle.load(
    open(f'{main_path}/dataset/training_data_{orient}.pickle', 'rb'))
labels_class = pickle.load(
    open(f'{main_path}/dataset/labels_{orient}.pickle', 'rb'))



### Specify the choosen model
model_name = GP_ResNet50   # GP_ResNet50, GP_UNet,

#change model name to a string
m_name = str(model_name).split(" ")

#the model's input channel
in_ch = 1



### Specify Training Parameters

# loss function
# if GPU is available set loss function to use GPU
loss_func = nn.CrossEntropyLoss()

#learning rate
lr = 0.0001

# optimizer
opt = torch.optim.Adam

# number of training iterations
epochs = 30


###################################################################################################



# check torch version
print("torch version:", torch.__version__)


# Empty GPU memory
torch.cuda.empty_cache()



#####################################Main part of the code############################################

"""## 1- the Dataset stored in pickle file format is already loaded
      2- store images in Xt, labels in yt and masks in  mt lists
"""

#Create lists to store data
Xt = []
features = None
labels = None
label = []
mt = []


#Store images in Xt, masks in mt and labels in yt iteratively
for features,masks in training_data_seg:
  Xt.append(features)
  mt.append(masks)

yt = list(labels_class)
"""# Check the whole dataset for Healthy brain images:
***Crucial in 2D models***
"""
j, num_class = healthy_brains(mt)



"""## Dataset split into training, validation, testing

Split the dataset for training using cross-validation method.

1- 70 % of images for training
2- 15% of images for validating
3- 15% of images for testing

1- Set random_state argument to most commonly used arbitrary numbers 42,0 for reproducability purposes of the train_test_split everytime the function is called.
2- Set stratify argument = label to ensure that relative class frequencies is approximately preserved after splitting into train, Val, Test datasets.
"""

# 70 % training, 15% validating, 15% testing
X_train, X_test, y_train, y_test, m_train, m_test  = train_test_split(Xt, yt, mt, test_size=0.3, random_state=42, shuffle=True, stratify=yt)  # 70% training, 30% testing
X_valid, X_test, y_valid, y_test, m_valid, m_test  = train_test_split(X_test, y_test, m_test, test_size=0.5, random_state=0, shuffle=True, stratify=y_test)  # split testing set into 50% validation , 50% testing


#Rempty the lists and arrays to free up RAM/Cache
Xt = None
yt = None
features = None
labels = None
label = None
mt = None
training_data_seg = None


#Creating real time training, validation and testing dataset using auguementation

###Check if j>1 before augumentation, resulting in 4 hot encoded labels else 3 hot encoded labels
print("The value of the healthy brain counter j:", j)

train_set = AugumentedDataset(j, X_train, m_train,  y_train)
valid_set = AugumentedDataset(j, X_valid, m_valid, y_valid)
test_set = AugumentedDataset(j, X_test, m_test, y_test)


#Total no of original dataset image samples in each set"
train_samples = len(X_train)
valid_samples = len(X_valid)
test_samples = len(X_test)

print(f"Number of original training samples: {train_samples}")
print(f"Number of original validation samples: {valid_samples}")
print(f"Number of original testing samples: {test_samples}")


#Total no of augumented dataset image samples in each set
print(f"Number of augmented training samples: {train_samples * aug}")
print(f"Number of augmented validation samples: {valid_samples * aug}")
print(f"Number of augmented testing samples: {test_samples* aug}")


#Creating a DataLoader for the three sets with enabled shuffling and drop_last enabled for data samples of certain orientation not divisable by batch size, drops last nonfitted batch
train_gen = DataLoader(train_set, batch_size=batch, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
valid_gen = DataLoader(valid_set, batch_size=batch, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
test_gen = DataLoader(test_set, batch_size=batch_test, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)


#Set the training on GPU if available otherwise on CPU
name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(name)

print("Device name is", name)




"""# **Specify the Used Model**

1- call model
2- set model to run on GPU or CPU based on availibility
3- Set all the pretrained weights to trainable by enabling every layer's parameters as true
"""


if model_name == GP_ResNet50:
  model = model_name(img_channel=in_ch, num_classes=num_class)

elif model_name == GP_UNet:
  model = model_name()


#after the model has been called, use the changed model name to a string m name when saving checkpoint and model
print("Model name is:", m_name[1])


# set model to run on GPU or CPU based on availibility
model.to(device)


# set all paramters as trainable
for params in model.parameters():
    params.requires_grad = True


# set all paramters of the model as trainable
for name, child in model.named_children():
  for name2, params in child.named_parameters():
    params.requires_grad = True


# function to Check the total number of parameters in a PyTorch model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


print(count_parameters(model))


# print the NN model's architecture
print(model)




"""## Set Training Configuration

1- use CrossEntropyLoss as our loss function
2- use ADAM optimizer initially with a 0.0001 learning rate
3- set the the number of training epochs
4- Create empty lists to store training and validations losses and accuracies
"""


###Training Parameters###

# loss function
# if GPU is available set loss function to use GPU
criterion =loss_func.to(device)

# optimizer
optimizer = opt(params=model.parameters(), lr=lr)

# number of training iterations
epochs = epochs



logs,losses = find_lr(train_gen, optimizer, model, criterion, device, batch, aug, num_class)   #plt.plot(logs[10:-5],losses[10:-5])
plt.savefig(
    f'{main_path}/Results/find_lr_{m_name[1]}_{orient}_{batch}b_{aug}a.png')
plt.show()

