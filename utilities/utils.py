import os
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
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
import nibabel as nib


def getValStat(data: list, percentile: float=99):
    vals = []
    for datum in data:
        vals += list(datum.flatten())
    return mean(vals), stdev(vals), np.percentile(np.array(vals), percentile)

class toTorchIO():
    def __call__(self, image, mask=None):
        dim = len(image.shape)
        sub = {"img": tio.ScalarImage(tensor=np.expand_dims(image,0) if dim==3 else np.expand_dims(np.expand_dims(image,0),-1))}
        if mask is not None:
            sub['mask'] = tio.LabelMap(tensor=np.expand_dims(mask,0) if dim==3 else np.expand_dims(np.expand_dims(mask,0),-1))
        return tio.Subject(**sub)

class fromTorchIO():
    def __call__(self, subject):
        if 'mask' in subject:
            return subject['img'][tio.DATA].squeeze(), subject['mask'][tio.DATA].squeeze()
        else:
            return subject['img'][tio.DATA].squeeze()


def result_analyser(y_pred, y_true, mask_pred, mask_true, classes, path, image, out_act):

    os.makedirs(path, exist_ok=True)
    path2 = path + "/nifti/"
    os.makedirs(path2, exist_ok=True)
    path3 = path2 + "/falseClass/"
    os.makedirs(path3, exist_ok=True)

    save_confusion_matrix(y_pred, y_true, classes, os.path.join(path, "cm.png"))

    classify_rprt = classification_report(y_true, y_pred, zero_division=1)
    Jindex = round(jaccard_score(y_true, y_pred, average='weighted'), 2)

    with open(os.path.join(path, "results.txt"), "a") as output:
        output.write("\n" + "Classification report : " + "\n" + str(classify_rprt) + "\n")
        output.write(f"Jaccard index = {str(Jindex)}" + "\n")

    if mask_pred is not None:
        dice_scores = []
        iou_scores = []

        #new adopted multiotsu + offset
        offset = 0.1
        rotate = False
        flip = False

        for i in range(len(mask_pred)):
            
            if rotate and flip:

                mask_pred = rotating(mask_pred, k=1, axes=(1, 0))               #rotate 90 deg to the right one time
                image = rotating(image, k=1, axes=(1, 0))                       #rotate 90 deg to the right one time
                mask_true = rotating(mask_true, k=1, axes=(1, 0))               #rotate 90 deg to the right one time

                mask_pred = flipping(mask_pred, axis=1)                         #axis = 1 flips horizontal
                image = flipping(image, axis=1)                                 #axis = 1 flips horizontal
                mask_true = flipping(mask_true, axis=1)                         #axis = 1 flips horizontal


            if y_true[i] == y_pred[i]:
                #nfiti save
                nib.save((nib.Nifti1Image(mask_pred[i], None)), path2 + f'{i}_{y_pred[i]}_{y_true[i]}_mask_pred.nii.gz')    
                nib.save((nib.Nifti1Image(mask_true[i], None)), path2 + f'{i}_{y_pred[i]}_{y_true[i]}_mask_true.nii.gz')    
                img = image[i]
                if img.shape[0] == 4:
                    img = np.transpose(img,(1, 2, 0))
                nib.save((nib.Nifti1Image(img, None)), path2 + f'{i}_{y_pred[i]}_{y_true[i]}_image.nii.gz')
            else:   #save missclassified ones as well
                #nfiti save 
                nib.save((nib.Nifti1Image(mask_pred[i], None)), path3 + f'{i}_{y_pred[i]}_{y_true[i]}_mask_pred.nii.gz')
                nib.save((nib.Nifti1Image(mask_true[i], None)), path3 + f'{i}_{y_pred[i]}_{y_true[i]}_mask_true.nii.gz')
                img = image[i]
                if img.shape[0] == 4:
                    img = np.transpose(img,(1, 2, 0))
                nib.save((nib.Nifti1Image(img, None)), path3 + f'{i}_{y_pred[i]}_{y_true[i]}_image.nii.gz')

            #raw heatmap
            hm = plt.imshow(mask_pred[i], cmap=plt.cm.RdBu)
            plt.colorbar(hm)
            plt.savefig(
                os.path.join(
                    path,
                    f"{str(i)}_{str(y_pred[i])}_{str(y_true[i])}_heatmap.png",
                )
            )

            plt.close()

            if out_act == "None":
                #suppressed heatmap
                mask_pred[i][mask_pred[i]<=0] = 0
                hm = plt.imshow(mask_pred[i], cmap=plt.cm.RdBu)
                plt.colorbar(hm)
                plt.savefig(
                    os.path.join(
                        path,
                        f"{str(i)}_{str(y_pred[i])}_{str(y_true[i])}sup_heatmap.png",
                    )
                )

                plt.close()

            value = threshold_multiotsu(mask_pred[i], classes=3)[-1]
            value = max(0.0, value-offset)
            m_pred = (mask_pred[i] > value).astype(np.float32)

            #older otsu no offset approach
            # thresh = threshold_otsu(mask_pred[i])
            # m_pred = (mask_pred[i] > thresh).astype(np.float32)

            dice, iou = segscores(m_pred, mask_true[i])
            dice_scores.append(round(dice, 2))
            iou_scores.append(round(iou, 2))
            mask_diff = create_diff_mask_binary(m_pred, mask_true[i])

            Image.fromarray(mask_diff).save(
                os.path.join(
                    path,
                    f"{str(i)}_{str(y_pred[i])}_{str(y_true[i])}_diff.png",
                )
            )

                    #############

    else:
        dice_scores = [-1]
        iou_scores = [-1]

    with open(os.path.join(path, "results.txt"), "a") as output:
        output.write(f"Dice (Mask) = {str(median(dice_scores))}" + "\n")
        output.write(f"IoU (Mask) = {str(median(iou_scores))}" + "\n")

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

def save_confusion_matrix(y_pred, y_true, LABELS, path):
    cm = confusion_matrix(y_pred, y_true)    
    df_cm = pd.DataFrame(cm, LABELS, LABELS)
    plt.figure(figsize = (13,10)) #(9,6) w,h  inches between two to get bet fit
    sns.set(font_scale=2.3) # Adjust to fit
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.savefig(path)
    #plt.show()
    plt.close()


def segscores(y_pred, y_true):
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    
    union = np.sum(y_true_f + y_pred_f)
    dice = (2. * intersection + 1) / (union + 1)

    union = union - intersection
    iou =  (intersection + 1) / (union + 1)
    return dice, iou

class Dice(torch.nn.Module):
    """
    Class used to get dice_loss and dice_score
    """

    def __init__(self, smooth=1):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        return dice_loss, dice_score


def flipping(data_squeeze, axis):
    """ Flips the image  """
    return np.flip(data_squeeze, axis)                                          
    
def rotating(data_squeeze, k, axes):
    """ Rotates the image 90 degrees to the left or right """
    return np.rot90(data_squeeze, k, axes)       
        
