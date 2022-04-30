"""## Data set augumentation

1- Real time data aguementation as mentioned before plus image normalization. 
2- One hot encoding the labels:
                  2.1 into three classes in case there is no healthy subjects
                  2.2 into four classes in case of finding healthy subjects
"""

#Essential imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np

##for bspline Augumentation
import gryds 

##For 2D Aug
import albumentations as A
from imgaug import augmenters as iaa
import cv2




## Data Augumentation class
class AugumentedDataset(Dataset):
  def __init__(self, j, images, masks, labels):

    # images
    self.X = images

    # masks
    self.m = masks

    # labels
    self.y = labels

    #Healthy brain counter
    self.j = j


  
    # Transformation for converting original image array to a tensor
    self.transform = transforms.Compose([
        transforms.ToPILImage(),                                          
        transforms.RandomRotation(330, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),  
        transforms.Normalize([0], [1])        #mean zero, std=1 normalize values and convert data range to [0,1]
    ])



  def __len__(self):
    # return length of image samples
    return len(self.X)

   
  def __getitem__(self, idx):


    # perform transformations on one instance of X
    # Original image as a tensor
    data = self.transform(self.X[idx])
    data_mask = self.transform(self.m[idx])    



    # one-hot encode the labels
    # our labels start from 1 rather than 0 the start of indexing in python
    #when no healthy brains are detected, get 3 one hot encoded labels 
    # 0: (1, 0, 0) 1:(0, 1, 0) 2:(0, 0, 1)  
    #when healthy brains are detected, get 4 one hot encoded labels, label 0=healthy
    # 0: (1, 0, 0, 0) 1:(0, 1, 0, 0) 2:(0, 1, 0, 0) 3:(0, 1, 0, 0) 

 
    # If health brains are present
    if j >= 1:

      if torch.max(data_mask) == 0:  #0
        #change the label to healthy = 0:([1., 0., 0., 0.])
        self.y[idx] = 0

      #create 4 hot encoded labels to include healthy labels
      labels = torch.zeros(4, dtype=torch.float32)
      labels[int(self.y[idx])] = 1.0
      # print(labels)



    #If not
    else:
      #create 3 hot encoded labels
      labels = torch.zeros(3, dtype=torch.float32)
      labels[int(self.y[idx])-1] = 1.0
      # print(labels)


    # Augmented image and corresponding label will be returned
    return labels, data

