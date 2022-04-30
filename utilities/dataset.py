import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import nibabel as nib


##for bspline Augumentation
# import gryds 

# ##For 2D Aug
# import albumentations as A
# from imgaug import augmenters as iaa
# import cv2
from  .Helper import Helper

class TumourDataset(Dataset):
    def __init__(self, contrast, images, masks, labels, maxinorm=False, transforms=None, trans=None):        
        self.X = images # images        
        self.m = masks # masks
        self.y = labels # labels
        self.maxinorm = maxinorm
        self.transforms = transforms
        self.contrast = contrast
        self.trans = trans

    def __len__(self):
        return len(self.X) # return length of image samples

    def __getitem__(self, idx):
        image = self.X[idx].astype(np.float32)
        mask = np.expand_dims(self.m[idx].astype(np.float32), 0)
        label = self.y[idx]

        if self.transforms:
            # for debuging purposes
            # main_path = "/mnt/public/hadya/master_Yassin/Brats2020/Dataset" 
            # images1 = nib.Nifti1Image(image, None)      #np.eye(4)
            # nib.save((nib.Nifti1Image(image, None)), main_path + f'/4Ch_experement_nii/BraTS20_Training_b4Trans_Test{idx}_Ch4.nii.gz') 

            image_trans = self.transforms(image)
            
            ##save some examples after transformation to check correctness of the operation
            # if self.trans is not None:  #only could occur in training subbset
            # image_numpy = np.transpose(image_trans,(1, 2, 0))
            # image_numpy = np.array(image_numpy)
            # images2 = nib.Nifti1Image(image_numpy, None)      #np.eye(4)
            # # nib.save((nib.Nifti1Image(image_numpy, None)), main_path + f'/4Ch_experement_nii/BraTS20_Training_afterTrans_Test{idx}_Ch4.nii.gz') 


        if self.maxinorm:

            if image_trans.max()==0.0:
                print(f"max of this array with indx {idx} is zero after transform, therefore this image won't be transformed")
                image = self.trans(image)   #only will convert np to Tensor 
                image /= image.max()
                if image.max()==0.0:
                    raise RuntimeError(f"max of this array with indx {idx} is zero after ToTensor Trans, this is not acceptable and will result in loss is nan")

            else:
                image_trans /= image_trans.max()
                image = image_trans
        
        return image, mask, label
