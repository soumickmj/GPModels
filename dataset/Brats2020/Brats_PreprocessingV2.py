# Brats2020_Dataset_Preparation

"""

This is the main module for the following processes:
1- saving Brats slices in 3D nifti volume after preprocessing(size check) and exclude empty slices
2- create a pickle for the nonempty slices of type dictionary to save the labels



"""
"""
__author__ = "Hadya Yassin"
Created on Thu Sep 15 10:00:00 2021

"""

# useful libraries

import glob
import os
from os.path import join as pjoin
import pickle
from re import A
from SimpleITK.SimpleITK import Mask

import numpy as np
import torch
import torchcomplex.nn.functional as cF  # install from https://github.com/soumickmj/pytorch-complex


#for Brats2020:
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import sys

#4 channels imgs
import cv2
import nibabel as nib


def interpWithTorchComplex(data, size, mode="sinc"):
      data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
      if mode == "sinc":
            data = cF.interpolate(data, size=size, mode=mode)
      else:
            data = cF.interpolate(data+1j, size=size, mode=mode).real
      return data.numpy().squeeze()


####Datset Preperation####

###########################Specify changable parameters#######################################
# specify used Dataset
Dataset = "Brats20"     # Brats20
# specify desired output orientation
orient = "Axi"  #All or Axi
amount = "full"      #full, half, 100, TEST
half_slices = True
process = 2          #process 1 or 2

##2-Brats2020 Dataset
main_path =  '/mnt/public/hadya/master_Yassin/Brats2020/Dataset'

if process == 1:
    source_root = main_path + f'/{amount}/{orient}/MICCAI_BraTS2020_TrainingData/*'
elif process == 2:     
    source_root = main_path + f'/{amount}/{orient}/NonEmpty_MICCAI_BraTS2020_TrainingData/*'


#save_2D_Img = main_path + f'/Final_2D_{amount}_BraTS2020_TrainingData/*/*'       #where the 2D images would be saved
label_path = main_path + f'/pickles/{amount}/{orient}/dict_labels.pickle'


ls_1 = ["t1", "t2", "t1ce", "flair","seg"] 


t1 = glob.glob(os.path.join(source_root,'*t1.nii.gz'), recursive=True)


#initiate lists for mask and rest of contrast data in ls_1
for m in range(len(ls_1) - 1):
    locals()[f"{ls_1[m+1]}"] = [0] * len(t1)


# Create list to store labels
labels = []
HG = 0
LG = 0
H = 0
e = 0
my_list = []
my_dict = {}


# for i in range(3):
for i in range(len(t1)):

    imgs = []
    msks = []
    t2_imgs = []
    t1ce_imgs = []
    flair_imgs = []
    X = []


    #to be able to save with correct name Ex when i = 190 the loaded file is 180 so to save i+1 will be wrong instead should be 180
    t1_mod = t1[i].replace("_", " ")
    numbers = [int(word) for word in t1_mod.split() if word.isdigit()]
    number = numbers[0]

    # print(i+1)
    print(number)
    # print("")


    if process == 2:
        # csv file name
        file_name = main_path + "/full/Axi/MICCAI_BraTS2020_TrainingData/name_mapping.csv"
        fields = ['Grade']
        df = pd.read_csv(file_name, skipinitialspace=True, usecols=fields)
        # See content in 'star_name'
        # print(df.Grade)
        if df.Grade[number-1] == "LGG":
            correct_name = "LGG"

        elif df.Grade[number-1] == "HGG":
            correct_name = "HGG"



    destin_root = main_path + f'/{amount}/{orient}/NonEmpty_MICCAI_BraTS2020_TrainingData/BraTS20_Training_00{number}'
    os.makedirs(destin_root, exist_ok=True)


    for p in range(len(ls_1) - 1):
        (locals()[f"{ls_1[p+1]}"])[i] = (locals()[f"{ls_1[0]}"])[i].replace(f"{ls_1[0]}", f"{ls_1[p+1]}")

    ls_2 = ["images", "t2_images", "t1ce_images", "flair_images", "masks"]
    for q in range(len(ls_2)):
        # images = nib.load(t1[i])
        locals()[f"{ls_2[q]}"] = nib.load((locals()[f"{ls_1[q]}"])[i])
        locals()[f"{ls_2[q]}"] = np.array(locals()[f"{ls_2[q]}"].dataobj)


    ls_3 = ["img", "t2_img", "t1ce_img", "flair_img", "mask"] 


    # for j in range(1,40):  # axi all volume [1-155]
    for j in range(images.shape[2]):  # axi all volume [1-155]

        for h in range(len(ls_3)):
            # img = images[:,:,j]
            locals()[f"{ls_3[h]}"] = (locals()[f"{ls_2[h]}"])[:,:,j]

        if locals()[f"{ls_3[0]}"].shape != (240, 240) or locals()[f"{ls_3[1]}"].shape != (240,240):
            img = interpWithTorchComplex(img, size=(240,240))
            mask = interpWithTorchComplex(mask, size=(240,240), mode="nearest")
            t2_img = interpWithTorchComplex(img, size=(240,240))
            t1ce_img = interpWithTorchComplex(img, size=(240,240))
            flair_img = interpWithTorchComplex(img, size=(240,240))

        mask[mask>0] = 1
        mask[mask<=0] = 0


        ls_4 = ["imgs", "t2_imgs", "t1ce_imgs", "flair_imgs", "msks"]  
        #note t1 image is represented as locals()[f"{ls_3[0]}"] = img from ls_3 list
        #note mask is represented as locals()[f"{ls_3[1]}"] = mask from ls_3 list

        ###excluding empty images and their empty masks
        
        #including tumour images and their masks
        if mask.any() > 0:   
            for d in range(len(ls_4)):
                (locals()[f"{ls_4[d]}"]).append(locals()[f"{ls_3[d]}"])

            if process == 2:
                if correct_name == "LGG":
                    label = 1.0
                    LG = LG + 1
                elif correct_name == "HGG":
                    label = 2.0
                    HG = HG + 1

                entry = {f"{number}_{j}": label}
                my_dict.update(entry)

        #including healthy images and their masks
        elif img.any() > 0:
            for d in range(len(ls_4)):
                (locals()[f"{ls_4[d]}"]).append(locals()[f"{ls_3[d]}"])

            if process == 2:
                label = 0.0
                H = H +1

                labels.append(label)
                entry = {f"{number}_{j}": label}
                my_dict.update(entry)

        else:
            e = e + 1
    
    if process == 1:
        ls_5 = ["t1_3D","t2_3D","t1ce_3D","flair_3D","seg_3D"] 

        for c in range(len(ls_5)):

            if ls_5[c] == "X_3D":
                locals()[f"{ls_5[c]}"] = np.zeros( (X[0].shape[0], X[0].shape[1],X[0].shape[2], len(X)) )
            else:
                locals()[f"{ls_5[c]}"] = np.zeros( (imgs[0].shape[0], imgs[0].shape[1], len(imgs)) )

            for w in range(len(imgs)):
                if ls_5[c] == "X_3D":
                    (locals()[f"{ls_5[c]}"])[:,:,:,w] = (locals()[f"{ls_4[c]}"])[w]
                else:
                    """Add the channels to the needed image one by one"""
                    (locals()[f"{ls_5[c]}"])[:,:,w] = (locals()[f"{ls_4[c]}"])[w]

            images = nib.Nifti1Image(locals()[f"{ls_5[c]}"], None)      #np.eye(4)
            nib.save(images, destin_root + f'/BraTS20_Training_00{number}_{ls_1[c]}.nii.gz')     #{}_{}.nii'.format(filename, pid)))
       


    print("Done")


print(f"total no of HGG slices: {HG}")
print(f"total no of LGG slices: {LG}")
print(f"total no of Healthy slices: {H}")
print(f"total no of empty slices: {e}")

if process == 2:

    sys.stdout = open(f"P3/{amount}/{orient}/output.txt", "w")

    #convert label list to array to use in pickle
    print("length of label dictionary is : ", len(my_dict))                #Shape must be (3064,) for all/ (1025,) for sag ...

    sys.stdout.close()

    pickle_out = open(label_path, "wb")
    pickle.dump(my_dict, pickle_out)
    pickle_out.close()

