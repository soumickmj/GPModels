# Post-Processing of the datasets

"""

This is the main module for the following processes:
1- main process, which allow to experiment with different thresholding techniques, calculate Dice scores and save the output images 
2- not main process, loads the saved lists to calculate dice scores mean, median and std dev. Additionally plotting the dice scores using a violin plot and a histogram



"""
"""
__author__ = "Hadya Yassin"
Created on Thu Aug 30 12:00:00 2021

"""

# useful libraries
import os
from posixpath import basename
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from numpy.lib.function_base import average
import pandas as pd
import seaborn as sns
import torch
import torchio as tio
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.filters import threshold_isodata
from skimage.filters import threshold_li
from skimage.filters import threshold_local
from skimage.filters import threshold_mean
from skimage.filters import threshold_minimum
from skimage.filters import threshold_multiotsu
from skimage.filters import threshold_niblack
from skimage.filters import threshold_sauvola
from skimage.filters import threshold_triangle
from skimage.filters import threshold_yen
from skimage.filters import try_all_threshold
from sklearn.metrics import (classification_report, confusion_matrix,
                             jaccard_score)


#added local threshholding of the bimodal histogram like otsu https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
import SimpleITK as sitk
import glob
import os
from os.path import join as pjoin
import pickle
import seaborn as sns
import nibabel as nib


def flipping(data_squeeze, axis):
    """ Flips the image  """
    return np.flip(data_squeeze, axis)                                          

def rotating(data_squeeze, k, axes):
    """ Rotates the image 90 degrees to the left or right """
    return np.rot90(data_squeeze, k, axes)       
        

def create_diff_mask_binary(predicted, label):
    """
    Method find the difference between the 2 binary images and overlay colors
    predicted, label : slices , 2D tensor
    """

    diff1 = np.subtract(label, predicted) > 0 # under_detected
    diff2 = np.subtract(predicted, label) > 0 # over_detected

    predicted = predicted > 0

    # Define colors
    red = np.array([255, 0, 0], dtype=np.uint8)  # under_detected
    green = np.array([0, 255, 0], dtype=np.uint8)  # over_detected
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output
    blue = np.array([0, 0, 255], dtype=np.uint8) # over_detected
    yellow = np.array([255, 255, 0], dtype=np.uint8)  # under_detected

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((*predicted.shape, 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted] = white
    rgb_image[diff1] = red
    rgb_image[diff2] = blue
    return rgb_image

def segscores(y_pred, y_true):
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    
    union = np.sum(y_true_f + y_pred_f)
    dice = (2. * intersection + 1) / (union + 1)

    union = union - intersection
    iou =  (intersection + 1) / (union + 1)
    return dice, iou


###########################Specify changable parameters#######################################
# main_path =  '/home/hyassin/hadyaRoot/mnt/public/hadya/master_Yassin/output/Results/Deterministic_Runs/'
main_path =  '/mnt/public/hadya/master_Yassin/output/Results/Deterministic_Runs/'
# main_path =  "/run/media/hyassin/New Volume/CentOS/Results/Server Results after new code/Deterministic_Runs"
Dataset = "China"   #China or Brats20
network = "GP_reconresnt"  #"GP_UNet" #"resnet18" # "GP_reconresnt"
out_act = "softmax" #softmax or None
orient = "Axi"      #All Sag Cor Axi
contrast = "allCont"     #t1, t1ce, t2, flair, allCont 
trial = True        #trial and error to see which offset is most suitable while inspecting results visually and after calculating dice values
choosen = 0.5       #choosen offset after trial and error 
main = False
rotate = True
flip = False
ISMRM = False

if ISMRM:
    ext = "/nifti" if network == "GP_reconresnt" else "/**/**/nifti"
    source_root = f"/run/media/hyassin/New Volume/CentOS/ISMRM/choosen/{Dataset}/{network}/{out_act}/**" + ext
elif Dataset == "China":
    source_root = main_path + f"/{Dataset}/{network}/{out_act}/TrainedOnPredselect/**/nifti"
elif Dataset == "Brats20":
    source_root = main_path + f"/{Dataset}/{orient}/{contrast}/{network}/{out_act}/**/nifti"


masks_pred = glob.glob(os.path.join(source_root,'*_pred.nii.gz'), recursive=True)
masks_true = glob.glob(os.path.join(source_root,'*_true.nii.gz'), recursive=True)
images = glob.glob(os.path.join(source_root,'*_image.nii.gz'), recursive=True)


##to save histogram and violin plots for dice scores
path3 = masks_pred[0].replace(f"nifti/0_0_0_mask_pred.nii.gz", "results/")
os.makedirs(path3, exist_ok=True)

#to save images to compare offsets
path_trial = path3 + "OffTrial/"
os.makedirs(path_trial, exist_ok=True)

#path to save processed images
path = path3 + "procplusoff/"
os.makedirs(path, exist_ok=True)




max_dice_scores_perimg = []
# offsets_T = [-0.3, -0.1, 0.1, 0.2]
offsets = []
dice_scores_ls = []
means = []
medians = []
counter = 0
counters = []
std_devs = []

# #to check selected range
# for j in range(len(offsets_T)):
#     offset =  offsets_T[j]

#to chcek a full range
for x in np.arange(0.4, 1.1, 0.1): #last number in range not included
    
    offset = round(x, 2)
    # print(f"offset = {offset}")

    ##to save dice values n lists
    path2 = source_root.replace(f"nifti", f"results/DiceValue/{offset}") 
    os.makedirs(path2, exist_ok=True)

    dice_scores = []
    dice_scores_dict = []
    iou_scores = []
    numbersls = []
    ls = ["numbersls", "dice_scores", "iou_scores", "dice_scores_dict"]


    if main:

        for i in range(100):
        # for i in range(len(masks_pred)):
            
            # #to get if the sample was classified correctly or not from the path name of the sample
            # mod = masks_pred[i].replace("_", " ")
            # values = [int(word) for word in mod.split() if word.isdigit()]
            # #values[0] = numbers[1]
            # #values[1] = numbers[2]

            #to be able to save with correct name Ex when i = 190 the loaded file is 180 so to save i+1 will be wrong instead should be 180
            t1_mod = masks_pred[i].replace(masks_pred[i], os.path.basename(masks_pred[i]))
            t1_mod = t1_mod.replace("_", " ")
            numbers = [int(word) for word in t1_mod.split() if word.isdigit()]
            number = numbers[0]
            numbersls.append(number)

            if numbers[1]!= 0 and numbers[2] != 0:

                #load nifti images
                image = ((nib.load(images[i])).get_fdata()).astype(np.float32)
                mask_pred = ((nib.load(masks_pred[i])).get_fdata()).astype(np.float32)
                mask_true = ((nib.load(masks_true[i])).get_fdata()).astype(np.float32)
                
                # #subplot(r,c) provide the no. of rows and columns
                # f, axarr = plt.subplots(1,3) 
                # # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                # axarr[0].imshow(image)
                # axarr[1].imshow(mask_pred)
                # axarr[2].imshow(mask_true)
                # f.tight_layout()
                # plt.show()
                

                if rotate:
                    image = rotating(image, k=1, axes=(1, 0))                       #rotate 90 deg to the right one time
                    mask_pred = rotating(mask_pred, k=1, axes=(1, 0))               #rotate 90 deg to the right one time
                    mask_true = rotating(mask_true, k=1, axes=(1, 0))               #rotate 90 deg to the right one time

                if flip:
                    image = flipping(image, axis=1)                                 #axis = 1 flips horizontal
                    mask_pred = flipping(mask_pred, axis=1)                         #axis = 1 flips horizontal
                    mask_true = flipping(mask_true, axis=1)                         #axis = 1 flips horizontal

                if offset == choosen or trial:
                    #get raw heatmap
                    hm1 = plt.imshow(mask_pred, cmap=plt.cm.RdBu)
                    plt.colorbar(hm1)
                    # plt.savefig(os.path.join(path, f"{number}_{numbers[1]}_{numbers[2]}_heatmap.png"))
                    # plt.show()
                    plt.close()

                # # to get suppressed heatmap
                mask_pred[mask_pred<=0] = 0

                offset2 = offset

                value = threshold_multiotsu(mask_pred, classes=3)[-1]
                value = max(0.0, value-offset2) 
                m_pred = (mask_pred > value).astype(np.float32)
                # plt.figure()
                # plt.imshow(m_pred)
                # plt.show()

                dice, iou = segscores(m_pred, mask_true)
                dice_scores.append(round(dice, 2))
                iou_scores.append(round(iou, 2))

                my_dict = {"number":[],"dice":[], "iou":[]};
                my_dict["number"].append(number)
                my_dict["dice"].append(dice)
                my_dict["iou"].append(iou)
                dice_scores_dict.append(my_dict)
                # print(f"image no : {number}, Dice = {dice}")
                # print(my_dict)
                # print(dice_scores_dict)

                # mask_diff = create_diff_mask_binary(m_pred, mask_true)
                # # plt.figure()
                # # plt.imshow(mask_diff)
                # # plt.show()


                # #subplot(r,c) provide the no. of rows and columns
                # f, axarr = plt.subplots(1,3) 
                # # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                # axarr[0].imshow(m_pred)
                # axarr[1].imshow(mask_true)
                # axarr[2].imshow(mask_diff)
                # f.tight_layout()
                # plt.show()


                if trial:
                    #save difference image
                    # Image.fromarray(mask_diff).save(os.path.join(path_trial, f"{number}_{numbers[1]}_{numbers[2]}_diff_off{offset2}.png"))
                    yu = 0
                
                elif offset2 == choosen:

                    ##Save input image and mask_true
                    plt.imsave(path + f"{number}_{numbers[1]}_{numbers[2]}_image.png", image, cmap='gray')
                    plt.imsave(path + f"{number}_{numbers[1]}_{numbers[2]}_mask_true.png", mask_true, cmap='gray')

                    #Save suppressed heatmap when not using softmax
                    if not out_act == "softmax":
                        hm2 = plt.imshow(mask_pred, cmap=plt.cm.RdBu)
                        plt.colorbar(hm2)
                        plt.savefig(os.path.join(path, f"{number}_{numbers[1]}_{numbers[2]}sup_heatmap_off{offset2}.png"))
                        # plt.show
                        plt.close()

                    #Save mask pred after thresholding using otsu
                    plt.imsave(path + f"{number}_{numbers[1]}_{numbers[2]}_mask_pred_after_otsu_off{offset2}.png", m_pred, cmap='gray')

                    #save difference image
                    Image.fromarray(mask_diff).save(os.path.join(path, f"{number}_{numbers[1]}_{numbers[2]}_diff_off{offset2}.png"))
                
            ##############

            ##in utils old Exploring different Threshhold modes  `https://scikit-image.org/docs/dev/api/skimage.filters.html
        
        #save the obtained lists for each offset
        for i in range(len(ls)):
            with open(path2 + f"/{ls[i]}.txt", "wb") as locals()["fp"+str(i+1)]:   # Unpickling
                pickle.dump(locals()[f"{ls[i]}"], locals()["fp"+str(i+1)])
        #save them all in one summary file for each offset
        with open(path2 + "/summary.txt", "w") as output:
            output.write(f"offset: {offset}, Median Dice: {median(dice_scores)} \n \n")
            output.write(f"numbers: \n {numbersls} \n \n")
            output.write(f"dice_scores: \n {dice_scores} \n \n")
            output.write(f"iou_Scores: \n {iou_scores} \n \n")
            output.write(f"Number_dice_iou dictlst \n {dice_scores_dict} \n \n")

        if offset == choosen:
            print(f"for {offset},  numbers: {numbersls}")
            print(f"for {offset},  dice scores: {dice_scores}")
            print(f"offset = {offset}, Median Dice (Mask) = {median(dice_scores)}, Mean Dice (Mask) = {mean(dice_scores)}, Std dev Dice (Mask) = {stdev(dice_scores)} \n")
            
            ###Plot Histogram   https://stackoverflow.com/questions/33203645/how-to-plot-a-histogram-using-matplotlib-in-python-with-a-list-of-data
            x = np.array(dice_scores)
            x = np.random.normal(size=1000)
            q25, q75 = np.percentile(x, [0.25, 0.75])
            bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
            bins = round((x.max() - x.min()) / bin_width)
            print("Freedman–Diaconis number of bins:", bins)
            plt.hist(x, bins=bins)
            plt.ylabel('No of iamges')
            plt.xlabel('DiceScore')
            plt.savefig(os.path.join(path3, f"DiceScore_Histogram_off-{offset}.png"))
            plt.show()
            plt.close()
            
            # plt.hist(x, density=True, bins=30)  # density=False would make counts
            # plt.ylabel('No of iamges')
            # plt.xlabel('DiceScore')
            # plt.show()

            print("Done")

    else:   #load dice scores list and calculate needed values

        # read lists
        # for i in range(len(ls)):
        i = 1 #load dice scores only
        with open(path2 + f"/{ls[i]}.txt", "rb") as locals()["fp"+str(i+1)]:   # Unpickling
            locals()[f"{ls[i]}"] = pickle.load(locals()["fp"+str(i+1)])
        
        for o in range(len(dice_scores)):
            if dice_scores[o] >= 0:
                counter += 1
        counters.append(counter)
        counter = 0

        Mean = mean(dice_scores)
        means.append(Mean)
        Median = median(dice_scores)
        medians.append(Median)
        Std_dev = stdev(dice_scores)
        std_devs.append(Std_dev)

        dice_scores_ls.append(dice_scores)
        offsets.append(offset)

        print(means)
        print(medians)
        print(std_devs)
        print(counters)

        #Plot dice scores in a Violin plot
        fig, ax = plt.subplots()
        ax.violinplot(dice_scores_ls, showmeans=True, showmedians=True)

        # add title and axis labels
        ax.set_title('violin plot')
        ax.set_xlabel('offsets')
        ax.set_ylabel('Dice scores')

        # add x-tick labels
        xticklabels = []
        xticks = []
        for h in range(len(offsets)):
            xticklabels.append(offsets[h])
            xticks.append(h+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # add horizontal grid lines
        ax.yaxis.grid(True)

        # save and show the plot
        plt.savefig(os.path.join(path3, f"DiceScore_ViolinPlot.png"))
        plt.show()

        

print("Done")
