#!/usr/bin/env python

"""

This is the main module of preprocesssing a given dataset:
1- Cropping the image
2- Zero Padding
3- Reorienting
4- flipping
5- Rotating
Individual operations as well as combined is available


"""
"""
__author__ = "Hadya Yassin"
Created on Thu Oct 17 21:09:39 2019

"""

# useful libraries
import os
import numpy as np
import nibabel as nib # import nibabel and load the image
import glob
# import tensorflow as tf




#############Delete unwanted files from dataset to run Siena-x (Delete everything but fully, recon, under)





class preprocessing:

    def cropping(self, data_squeeze, IMAGE_HEIGHT, IMAGE_WIDTH):
        """ Crops the image to the desired size """
        # return tf.image.resize_with_crop_or_pad(data_squeeze, IMAGE_HEIGHT, IMAGE_WIDTH)  
      

    def padding(self, data_squeeze, pad_width, mode):
        """ Zero pads the image to the desired size """
        return np.pad(data_squeeze, pad_width, mode)                                                             

   
    def reorienting(self, data_squeeze, axes):
        """ Reorient image to (Sagittal, Coronal, Axial) """
        return np.transpose(data_squeeze, axes)                       


    def flipping(self, data_squeeze, axis):
        """ Flips the image  """
        return np.flip(data_squeeze, axis)                                          


    def rotating(self, data_squeeze, k, axes):
        """ Rotates the image 90 degrees to the left or right """
        return np.rot90(data_squeeze, k, axes)       
        

    def delete(self, folderPath):
        """ Deletes unwanted files in the folder """
        files = glob.glob(os.path.join(folderPath,'**/**/*.txt')) + glob.glob(os.path.join(folderPath,'**/**/reconCorrected.nii')) + glob.glob(os.path.join(folderPath,'**/**/reconCorrectedNorm.nii'))

        for file in files:
            os.remove(file)



###########################Specify changable parameters#######################################
# specify used Dataset
Dataset = "Brats20"     # Brats20
# specify desired output orientation
orient = "Axi"       #original orientation of Brats is Axi
amount = "full"       #full, half, 100, TEST
added = "_reoriented"
contrast = "flair"
mix_contrast = True  #Preprocessing all type of contrast together or mix

main_path =  '/mnt/public/hadya/master_Yassin/Brats2020/Dataset'
source_root = main_path + f'/{amount}/{orient}/NonEmpty_MICCAI_BraTS2020_TrainingData/*'

if not mix_contrast:
    files = glob.glob(os.path.join(source_root,f'*{contrast}.nii.gz'), recursive=True)

#specify the mixed contrast
elif mix_contrast:
    contrast = "t1"
    added = "allCont_reoriented"
    files = glob.glob(os.path.join(source_root,f'*{contrast}.nii.gz'), recursive=True)


seg = [0] * len(files)

# Specify the preprocessing needed by "True" or "False"
crop = False      
pad = False
reorient = True
flip = False
rotate = False

h = 0

for file in files:

    if not mix_contrast:
        data_squeezes = [0] * 2
        affines = [0] * 2
        seg = file.replace(f"{contrast}", "seg")  
        img = nib.load(file)
        data_squeezes[0] = np.squeeze(img.get_fdata(), axis=None)
        affines[0] = img.affine
        mask = nib.load(seg)
        data_squeezes[1] = np.squeeze(mask.get_fdata(), axis=None)
        affines[1] = mask.affine
        

    elif mix_contrast:
        con = ["t1", "seg", "t1ce", "t2", "flair"]
        data_squeezes = [0] * len(con)
        affines = [0] * len(con)

        for i in range(len(con)):
    

            if con[i] == contrast:
                # destin_path = destin_path
                img = nib.load(file)
                data_squeezes[i] = np.squeeze(img.get_fdata(), axis=None)
                affines[i] = img.affine

            else:
                # destin_path = destin_path.replace(f"{contrast}", f"{con[i]}") 
                img = nib.load(file.replace(f"{contrast}", f"{con[i]}") )
                data_squeezes[i] = np.squeeze(img.get_fdata(), axis=None)
                affines[i] = img.affine



    t = preprocessing()

    if not mix_contrast:
        if crop:
            #specify arguments values:
            data_squeezes[0] = t.cropping(data_squeezes[0], IMAGE_HEIGHT = 240 ,IMAGE_WIDTH = 240)                          #crops image to 240x240 
            data_squeezes[1] = t.cropping(data_squeezes[1], IMAGE_HEIGHT = 240 ,IMAGE_WIDTH = 240)                          #crops image to 240x240 
        
        
        if pad:
            #specify arguments values:
            data_squeezes[0] = t.cropping(data_squeezes[0], IMAGE_HEIGHT = 240 ,IMAGE_WIDTH = 240)                          #crops image to 240x240 
            data_squeezes[1] = t.padding(data_squeezes[1], pad_width = [(0,0), (8,8), (50,50)], mode = 'constant')          #zeropads images adds zeros left and right od each img dimension


        if reorient:
            #specify arguments values:
            # provided that an axial input image with axes = (0, 1, 2) is given

            h = h+1

            if h == 1:
                print("Axial")
                orientation = "Axi"
            elif h == 2:
                data_squeezes[0] = t.reorienting(data_squeezes[0], axes=(2, 1, 0))   # Sagittal axes=(2, 1, 0) = axes=None = axes=range(2, -1, -1)     
                data_squeezes[1] = t.reorienting(data_squeezes[1], axes=(2, 1, 0))   # Sagittal axes=(2, 1, 0) = axes=None = axes=range(2, -1, -1)     
                print ("Sagittal")
                orientation = "Sag"
                data_squeezes[0] = t.rotating(data_squeezes[0], k=1, axes=(1, 0))    #rotate img 90 to left to get std                                            #rotate 90 deg to the right one time
                data_squeezes[1] = t.rotating(data_squeezes[1], k=1, axes=(1, 0))    #rotate mask 90 to left to get std like image  
            elif h == 3:
                data_squeezes[0] = t.reorienting(data_squeezes[0], axes=(2, 0, 1))   # Coronal axes=(2, 1, 0)
                data_squeezes[1] = t.reorienting(data_squeezes[1], axes=(2, 0, 1))   # Coronal axes=(2, 1, 0)
                print ("Coronal")
                orientation = "Cor"
                data_squeezes[0] = t.rotating(data_squeezes[0], k=1, axes=(1, 0))    #rotate img 90 to left to get std                                            #rotate 90 deg to the right one time
                data_squeezes[1] = t.rotating(data_squeezes[1], k=1, axes=(1, 0))    #rotate mask 90 to left to get std like image            
                h = 0                                         

        if flip:
            #specify arguments values:
            data_squeezes[0] = t.flipping(data_squeezes[0], axis=1)                                                         #axis = 1 flips horizontal
            data_squeezes[1] = t.flipping(data_squeezes[1], axis=1)                                                         #axis = 1 flips horizontal


        if rotate:
            #specify arguments values:
            data_squeezes[0] = t.rotating(data_squeezes[0], k=1, axes=(0, 1))                                               #rotate 90 deg to the right one time
            data_squeezes[1] = t.rotating(data_squeezes[1], k=1, axes=(0, 1))                                               #rotate 90 deg to the right one time

    elif mix_contrast:

        if reorient:
            #specify arguments values:
            # provided that an axial input image with axes = (0, 1, 2) is given

            h = h+1

            if h == 1:
                print("Axial")
                orientation = "Axi"
            
            elif h == 2:
                for j in range(len(con)): 
                    data_squeezes[j] = t.reorienting(data_squeezes[j], axes=(2, 1, 0))   # Sagittal axes=(2, 1, 0) = axes=None = axes=range(2, -1, -1)      
                    print ("Sagittal")
                    orientation = "Sag"
                    data_squeezes[j] = t.rotating(data_squeezes[j], k=1, axes=(1, 0))    #rotate img 90 to left to get std                                            #rotate 90 deg to the right one time
            
            elif h == 3:
                for j in range(len(con)): 
                    data_squeezes[j] = t.reorienting(data_squeezes[j], axes=(2, 0, 1))   # Coronal axes=(2, 1, 0)
                    print ("Coronal")
                    orientation = "Cor"
                    data_squeezes[j] = t.rotating(data_squeezes[j], k=1, axes=(1, 0))    #rotate img 90 to left to get std                                            #rotate 90 deg to the right one time
                h = 0                                         


    if Dataset == "Brats20" and orient == "All":
        destin_path = file.replace(f'/{amount}_Axi_MICCAI_BraTS2020_TrainingData',f'/{amount}_{contrast}{added}_{orient}_MICCAI_BraTS2020_TrainingData/{orientation}')
        
        if not mix_contrast:
            destin_paths = [0] * 2
            destin_path5 = seg.replace(f'/{amount}_Axi_MICCAI_BraTS2020_TrainingData',f'/{amount}_{contrast}{added}_{orient}_MICCAI_BraTS2020_TrainingData/{orientation}')
            destin_paths[0] = destin_path
            destin_paths[1] = destin_path5

        elif mix_contrast:
            destin_path = destin_path.replace(f'{contrast}allCont','allCont')
            destin_paths = [0]* len(con)
            for k in range(len(con)):
        
                if con[k] == contrast:
                    destin_paths[k] = destin_path
                else:
                    destin_paths[k] = destin_path.replace(f"{contrast}", f"{con[k]}") 



    os.makedirs(destin_paths[0].replace(os.path.basename(destin_paths[0]), '')[:-1],exist_ok=True)
    
    for r in range(len(destin_paths)): 
        nib.save(nib.Nifti1Image(data_squeezes[r], affines[r]), destin_paths[r]) 




#optional(Delete unwanted files in reconstructed output files):
delete = "False"
folderPath = r'C:\Final\enterprisencc1701_hadya\Brain-tumor-segmentation-master\Brain-tumor-segmentation-master\BratsValidationSet_ReconstructedOutput\h7'
if delete == "True":
    #specify arguments values:
    t = preprocessing()
    t.delete(folderPath)  


print('Done')