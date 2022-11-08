import os
import pickle
import sys
import glob
import nibabel as nib
import numpy as np
from torchvision import transforms
import torch
import random
import torchio as tio
from utilities.utils import (fromTorchIO, toTorchIO)


def load_Brats_full(amount, orient, main_path, contrast):

    Test = False

    label_path = f'{main_path}/pickles/{amount}/{orient}/dict_labels.pickle'
    labels = pickle.load(open(label_path, 'rb'))

    in_channel = (4 if contrast == "allCont" else 1)

    # Create list to store labels
    y = []
    X = []
    m = []

    idx = 0     #idx used for all images stacked on top one anothe in one volume returned at the end = X


    HG = 0
    LG = 0
    H = 0

    source_root = f'{main_path}/{amount}/{orient}/NonEmpty_MICCAI_BraTS2020_TrainingData/*'


    t1 = glob.glob(os.path.join(source_root,'*t1.nii.gz'), recursive=True)

    if in_channel == 4:
        ls_1 = ["t1", "t2", "t1ce", "flair", "seg"]
        #initiate lists for mask and rest of contrast data in ls_1
        for n in range(len(ls_1) - 1):
            locals()[f"{ls_1[n+1]}"] = [0] * len(t1)

    else:
        if contrast != "t1":
            locals()[f"{contrast}"] = [0] * len(t1)
        Seg = [0] * len(t1)


    ls_2 = ["t1_images", "t2_images", "t1ce_images", "flair_images", "masks"]

    ls_3 = ["t1_img", "t2_img", "t1ce_img", "flair_img", "mask"]

    for i in range(3 if Test else len(t1)):

        #to be able to save with correct name Ex when i = 190 the loaded file is 180 so to save i+1 will be wrong instead should be 180
        t1_mod = t1[i].replace("_", " ")
        numbers = [int(word) for word in t1_mod.split() if word.isdigit()]
        number = numbers[0]

        if in_channel == 4:
            for p in range(len(ls_1) - 1):
                (locals()[f"{ls_1[p+1]}"])[i] = (locals()[f"{ls_1[0]}"])[i].replace(f"{ls_1[0]}", f"{ls_1[p+1]}")

            for q in range(len(ls_2)):
                locals()[f"{ls_2[q]}"] = ((nib.load((locals()[f"{ls_1[q]}"])[i])).get_fdata()).astype(np.float32)

            no_slices = locals()[f"{ls_2[0]}"].shape[2]

        else:
            if contrast != "t1":
                (locals()[f"{contrast}"])[i] = t1[i].replace("t1", f"{contrast}")

            Seg[i] = t1[i].replace("t1", "seg")

            locals()[f"{contrast}_images"] = ((nib.load((locals()[f"{contrast}"])[i])).get_fdata()).astype(np.float32)
            ((nib.load((locals()[f"{ls_1[q]}"])[i])).get_fdata()).astype(np.float32)

            Masks = ((nib.load(Seg[i])).get_fdata()).astype(np.float32)

            no_slices = locals()[f"{contrast}_images"].shape[2]

        ll = 0

        # for j in range(1,156):  # axi all volume [1-155]
        for j in range(no_slices):  # axi all volume [1-155]
            
            if in_channel == 4:
                #get a slice from each contrast
                for h in range(len(ls_3)):
                    locals()[f"{ls_3[h]}"] = (locals()[f"{ls_2[h]}"])[:,:,j]

                #1 stack the slices into a 4-ch image
                x = np.stack((locals()[f"{ls_3[0]}"], locals()[f"{ls_3[1]}"], locals()[f"{ls_3[2]}"], locals()[f"{ls_3[3]}"]), axis=-1)
                ##axes=-1 to get (H,W,C) format

                #save some examples before and after transformation could be done in dataset.py

                #check if image is empty
                if x.max() == 0.0:
                    print(f"max of this array with indx {i}_{j} is zero")
                    images_b1 = nib.Nifti1Image(x, None)
                    nib.save(
                        images_b1,
                        f'{main_path}/4Ch_experement_nii/BraTS20_Training_emptyimage_{j}_{h}_Ch4.nii.gz',
                    )


                else:   #Stack 4-ch images and masks
                    X.append(x) 
                    m.append(locals()[f"{ls_3[4]}"])
                    # print(idx)
                    idx += 1


            else:    #in_channel != 4
                locals()[f"{contrast}_img"] = (locals()[f"{contrast}_images"])[:,:,j]
                Mask = Masks[:,:,j]

                # if np.any(locals()[f"{contrast}_img"]) > 0 and (locals()[f"{contrast}_img"]).max() > 0:
                #     print(i,j)
                # elif (locals()[f"{contrast}_img"]).max() == 0:
                #     print(f"max of this {i}{j} array is zero")

                X.append(locals()[f"{contrast}_img"])  
                m.append(Mask)    
                # print(idx)
                idx += 1

            #append labels that are not for empty images after transform
            if not Test and x.max() != 0.0:
                y.append(labels[f"{number}_{j}"])
            elif not Test and x.max() == 0.0:
                print(f"ignore this label cause the image array with this indx {i}_{j} is zero")
            else:
                # Pseudo labels when testing with smaller volume
                ll += 1
                if ll == 1:
                    label = 0.0
                elif ll == 2:
                    label = 1.0
                elif ll == 3:
                    label = 2.0
                    ll = 0

                y.append(label)


            if int(labels[f"{number}_{j}"]) == 0:
                H =+ 1
            elif int(labels[f"{number}_{j}"]) == 1:
                LG =+ 1
            elif int(labels[f"{number}_{j}"]) == 2:
                HG =+ 1

            # if in_channel == 4:
            #     # save 4 ch volumes
            #     kk = np.zeros((100, 240, 240, 4))
            #     for jj in range(100):
            #         kk[jj,:,:,:] = X[jj]
            #     b = np.transpose(kk, (1, 2, 3, 0))
            #     images = nib.Nifti1Image(b, None)      #np.eye(4)
            #     nib.save(images, main_path + f'/BraTS20_Training_00{number}_Ch4.nii.gz') 

    #convert label list to array to use in pickle
    y = np.array(y, dtype=np.int64)
    # print("shape of label array is : ", y.shape)                #Shape must be (3064,) for all/ (1025,) for sag ...
    X = np.array(X)
    m = np.array(m) 

    for i in range(X.shape[0]):
        image = X[i].astype(np.float32)
        if image.max() == 0.0:
            print(f"Not acceptable to have a max of this array with indx {i} as zero, will result in loss is nan in dataset.py norm because / by 0")

    return X, m, y


