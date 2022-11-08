# Dataset Preparation

"""

This is the main module of create training_data of a certain orientation
as a pickle file of imgs and thei labels


"""
"""
__author__ = "Hadya Yassin"
Created on Thu Jan 01 18:00:00 2021

"""

# useful libraries

import glob
import os
import pickle

import h5py
import numpy as np
from tqdm import tqdm
import torch
import torchcomplex.nn.functional as cF

def interpWithTorchComplex(data, size, mode="sinc"):
      data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
      if mode == "sinc":
            data = cF.interpolate(data, size=size, mode=mode)
      else:
            data = cF.interpolate(data+1j, size=size, mode=mode).real
      return data.numpy().squeeze()

####Datset Preperation####


###########################Specify changable parameters#######################################

# main_path = '/run/media/hyassin/New Volume/CentOS/SourceTree_master_Yassin/seg-con-classify/dataset' 
main_path = r'D:\Rough\H\brainTumorDataPublic_China' 

# specify desired output orientation
orient = "All"       #All, Sag, Cor, Axi


# source_root = main_path + f'/{orient}matfiles'
source_root = f'{main_path}/MATs'
destin_root = f'{main_path}/Processed'
label_path = f'{main_path}/labels_{orient}.pickle'
Train_path = f'{main_path}/training_data_{orient}.pickle'

###################################################################################################


files = glob.glob(os.path.join(source_root,'*.mat'), recursive=True)


# Create list to store labels
labels = []
filenames = []
training_data_seg = []


c = 0  #counter

for file in tqdm(files):
      #extract file name from path
      filename = os.path.basename(file)
      filename = filename.replace(".mat", "")

      with h5py.File(file, 'r') as f:

            img = f['cjdata']['image']
            label = f['cjdata']['label'][0][0]
            # tumorBorder = f['cjdata']['tumorBorder'][0]  #not used
            mask = f['cjdata']['tumorMask']
            #pid = f['cjdata']['PID']                    #Patient ID      

            # # Creating training data labels for classification
            labels.append(int(label))
            img = np.array(img, dtype=np.float32)
            mask = np.array(mask, dtype=np.float32)

            if img.shape[0] != 512 or img.shape[1] != 512:
                  c = c + 1
                  img = interpWithTorchComplex(img, size=(512,512))
                  mask = interpWithTorchComplex(mask, size=(512,512), mode="nearest")
                  mask[mask>0] = 1
                  mask[mask<=0] = 0

            # # Creating training data set for Classification/segmentaton
            training_data_seg.append([img, mask])   


print(f"{len(files)} files successfully processed")
print(f"{c} files has off size and have been interpolated")

# #1- Creating pickle file for labels:

#convert label list to array to use in pickle
label_array = np.array(labels, dtype=np.int64)
print("shape of label array is : ", label_array.shape)                #Shape must be (3064,) for all/ (1025,) for sag ...


with open(label_path, "wb") as pickle_out:
      pickle.dump(label_array, pickle_out)
###2- Creating training_data as pickle file for images and their masks:
print(f"Shape of training data is : {str(np.shape(training_data_seg))}")


with open(Train_path, "wb") as pickle_out:
      pickle.dump(np.array(training_data_seg), pickle_out)
print('Done')


