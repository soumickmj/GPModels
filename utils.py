import os
from pathlib import Path
from statistics import median

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk
from skimage.measure import label, regionprops

from statistics import mean, stdev

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

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


def getValStat(data: list, percentile: float=99):
    vals = []
    for datum in data:
        vals += list(datum.flatten())
    return mean(vals), stdev(vals), np.percentile(np.array(vals), percentile)

def save_fold_predictions(y_true, y_prob, out_dir, fold_id):
    """
    Save per-sample probabilities and labels for one fold.
    y_true: 1D numpy array of shape (N,)
    y_prob: 2D numpy array of shape (N, C)
    out_dir: directory where the experiment results live (e.g. out_path/Results/trainID)
    fold_id: integer fold index
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"fold_{fold_id}_preds.npz",
        y_true=y_true.astype(int),
        y_prob=y_prob.astype(float),
    )
##################

def segscores(y_pred, y_true):
    """
    Dice + IoU for binary masks.
    y_pred, y_true float or uint8 in {0,1}.
    """
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    union = np.sum(y_true_f + y_pred_f)
    dice = (2.0 * intersection + 1.0) / (union + 1.0)

    union_minus_intersection = union - intersection
    iou = (intersection + 1.0) / (union_minus_intersection + 1.0)
    return float(dice), float(iou)



def save_confusion_matrix(y_pred, y_true, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    sns.set_theme(style="white", font_scale=2.2)

    plt.figure(figsize=(9, 7))   

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="rocket_r",
        square=True,
        linewidths=0,
        annot_kws={"size": 22},   
        cbar_kws={"shrink": 0.85},
        vmin=0,          
        vmax=6000      
    )

    # Axis labels
    ax.set_xlabel("Prediction", fontsize=24, labelpad=12)
    ax.set_ylabel("Target", fontsize=24, labelpad=12)

    # Tick labels
    ax.set_xticklabels(labels, rotation=0, fontsize=22)
    ax.set_yticklabels(labels, rotation=90, fontsize=22, va="center")

    # Colorbar tick size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)


    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_diff_mask_binary(predicted_bin, label_bin):
    """
    predicted_bin, label_bin: binary (0/1) arrays (H,W)
    returns RGB uint8 mask:
      white = predicted
      red   = missed (GT=1, pred=0)
      blue  = false positive (GT=0, pred=1)
    """
    predicted = predicted_bin.astype(bool)
    label = label_bin.astype(bool)

    under = np.logical_and(label, np.logical_not(predicted))
    over = np.logical_and(predicted, np.logical_not(label))

    rgb = np.zeros((*predicted.shape, 3), dtype=np.uint8)  # black background
    rgb[predicted] = np.array([255, 255, 255], dtype=np.uint8)  # white
    rgb[under] = np.array([255, 0, 0], dtype=np.uint8)          # red
    rgb[over] = np.array([0, 0, 255], dtype=np.uint8)           # blue
    return rgb


def heatmap_stats(hm: np.ndarray):
    hm = np.asarray(hm)
    return {
        "min": float(np.min(hm)),
        "max": float(np.max(hm)),
        "p50": float(np.percentile(hm, 50)),
        "p90": float(np.percentile(hm, 90)),
        "p95": float(np.percentile(hm, 95)),
        "p99": float(np.percentile(hm, 99)),
        "unique": int(np.unique(hm).size),
    }


def keep_topk_lcc(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """Keep largest (k=1) or top-k connected components in a binary mask."""
    lab = label(mask.astype(bool))
    if lab.max() == 0:
        return mask.astype(bool)

    props = regionprops(lab)
    props = sorted(props, key=lambda r: r.area, reverse=True)
    keep = props[:k]

    out = np.zeros_like(mask, dtype=bool)
    for r in keep:
        out[lab == r.label] = True
    return out


def threshold_heatmap_to_mask(       
    hm_raw: np.ndarray,
    out_act: str = "None",       
    prefer: str = "auto",        # "auto" | "fixed" | "otsu" | "multiotsu"
    fixed_thr: float = 0.5,
    multiotsu_classes: int = 3,
    multiotsu_nbins: int = 256,
    offset: float = 0.1,
    normalize: bool = True,
    network: object = None,  
):
    """
    Converts 2D heatmap -> binary mask + info dict.
    - suppress negatives to 0
    - optional normalize by max to [0,1]
    - auto: use fixed if prob-like else otsu
    - multiotsu supported with fallback
    """
    info = {}

    # hm = np.maximum(np.asarray(hm_raw).copy(), 0.0)

    # if normalize:
    #     mx = float(hm.max())
    #     hm = hm / mx if mx > 0 else hm * 0.0

    hm=hm_raw

    st = heatmap_stats(hm)
    info.update(st)

    method = prefer
    if method == "auto":
        # If the network output is already probability-like, fixed threshold is stable.
        # If it's logits/unnormalized, Otsu is safer.
        if str(out_act).lower() in ("sigmoid", "softmax") and not "MProto" in network:
            method = "fixed"
        # elif "Recon" in network:
        #     method = "fixed"
        else:    
            method = "multiotsu"

    thr = None
    if method == "fixed":
        thr = float(fixed_thr)

    elif method == "multiotsu":

        if "UNet" in network:  # GP_UNet
            offset=0.3
        elif "Recon" in network:  # GP_ReconResNet
            offset=1.6
        elif "Shuffle" in network:  # GP_ShuffleUNet    
            offset=0.1
        elif "MProto" in network:
            offset=0.5


        try:
            if st["unique"] >= multiotsu_classes:
                thr = threshold_multiotsu(
                    hm, classes=multiotsu_classes, nbins=multiotsu_nbins
                )[-1]   ##Strict thresh
                thr = max(0.0, float(thr) - float(offset))
            else:
                raise ValueError("not enough unique values for multiotsu")
        except Exception as e:
            info["fallback_from_multiotsu"] = str(e)
            method = "otsu"

    if method == "otsu":
        thr = float(threshold_otsu(hm))


    mask = (hm > thr).astype(np.uint8)
    info["method"] = method
    info["thr"] = float(thr)
    info["out_act"] = str(out_act)

    return mask, info, hm  # return normalized suppressed heatmap too


def clean_binary_mask(
    mask: np.ndarray,
    min_obj_px: int = 64,
    hole_area_px: int = 128,
    closing_radius: int = 2,
    keep_k: int = 1,
):
    """Remove speckles, fill holes, close, keep largest CC."""
    m = mask.astype(bool)
    m = remove_small_objects(m, min_size=int(min_obj_px))
    m = remove_small_holes(m, area_threshold=int(hole_area_px))
    if closing_radius and closing_radius > 0:
        m = binary_closing(m, footprint=disk(int(closing_radius)))
    if keep_k and keep_k > 0:
        m = keep_topk_lcc(m, k=int(keep_k))
    return m.astype(np.uint8)


def pick_examples(
    df: pd.DataFrame,
    K_worst: int = 30,
    K_best: int = 30,
    K_misclf: int = 60,
    K_fp_healthy: int = 60,
):
    # tumour slices (GT tumour present)
    tum = df[df["use_for_tumour_dice"] == True].copy()

    # misclassified classification: prefer those with tumour GT, then fill
    mis = df[df["y_true"] != df["y_pred"]].copy()
    mis["is_tumour_gt"] = (mis["gt_sum"] > 0).astype(int)
    mis = mis.sort_values(
        by=["is_tumour_gt", "dice_clean", "pred_sum_clean"],
        ascending=[False, True, False],
        na_position="last",
    )
    mis_pick = mis.head(K_misclf)



    # worst/best per class by dice_clean
    worst_lgg = tum[tum["y_true"] == 1].nsmallest(K_worst, "dice_clean")
    worst_hgg = tum[tum["y_true"] == 2].nsmallest(K_worst, "dice_clean")
    best_lgg  = tum[tum["y_true"] == 1].nlargest(K_best, "dice_clean")
    best_hgg  = tum[tum["y_true"] == 2].nlargest(K_best, "dice_clean")

    # healthy FP segmentation: GT empty but predicted blob exists
    healthy_fp = df[
        (df["y_true"] == 0) & (df["gt_sum"] == 0) & (df["pred_sum_clean"] > 0)
    ].copy()
    healthy_fp = healthy_fp.nlargest(K_fp_healthy, "pred_sum_clean")

    picked = pd.concat([mis_pick, worst_lgg, worst_hgg, best_lgg, best_hgg, healthy_fp]).drop_duplicates()
    return sorted(picked["slice_idx"].astype(int).tolist())



# Optional: save a few PNG panels + optional nifti
def save_slice_outputs(
    i: int,
    y_pred,
    y_true,
    mask_pred,
    mask_true,
    image,
    offset, 
    out_act: str,
    out_dir: str,
    save_nifti: bool = False,
    network: object = None,

):
    """
    Saves:
      - raw heatmap png
      - suppressed+normalized heatmap png
      - diff overlay png (using CLEAN mask)
      - (optional) nifti for image + pred mask + gt mask
    """
    ensure_dir(out_dir)
    nifti_dir = os.path.join(out_dir, "nifti")
    if save_nifti:
        ensure_dir(nifti_dir)

    # Heatmap raw (no suppression)
    plt.imshow(np.squeeze(mask_pred[i]), cmap=plt.cm.RdBu)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(out_dir, f"{i}_{y_pred[i]}_{y_true[i]}_heatmap.png"), dpi=150)
    plt.close()

    hm = np.squeeze(mask_pred[i])

    raw_mask, info, hm = threshold_heatmap_to_mask(
        hm_raw=hm,
        out_act=out_act,   
        prefer="auto",
        fixed_thr=0.5,
        multiotsu_classes=3,
        offset=offset,
        normalize=True,
        network=network,
    )

    hm_sup = np.maximum(hm, 0)  # negatives -> 0
    hm_sup = np.squeeze(hm_sup)

    plt.imshow(hm_sup, cmap=plt.cm.RdBu)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(out_dir, f"{i}_{y_pred[i]}_{y_true[i]}_sup_heatmap.png"), dpi=150)
    plt.close()

    # Clean mask + diff overlay
    clean_mask = clean_binary_mask(raw_mask, min_obj_px=64, hole_area_px=128, closing_radius=2, keep_k=1)
    gt = np.squeeze((mask_true[i] > 0).astype(np.uint8))

    diff_rgb = create_diff_mask_binary(clean_mask, gt)
    Image.fromarray(diff_rgb).save(os.path.join(out_dir, f"{i}_{y_pred[i]}_{y_true[i]}_diff.png"))

    flair = image[i][3, :, :]   # FLAIR
    # Normalize to uint8 for PNG
    flair = flair.astype(np.float32)
    mn, mx = flair.min(), flair.max()
    flair = (flair - mn) / (mx - mn + 1e-8)
    flair = (flair * 255).astype(np.uint8)
    Image.fromarray(flair).save(os.path.join(out_dir, f"{i}_{y_pred[i]}_{y_true[i]}_image.png"))
    Image.fromarray((raw_mask * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{i}_{y_pred[i]}_{y_true[i]}_rawnmask_strict.png"))
    Image.fromarray((clean_mask * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{i}_{y_pred[i]}_{y_true[i]}_cleanmask_strict.png"))
    Image.fromarray((gt * 255).astype(np.uint8)).save(os.path.join(out_dir, f"{i}_{y_pred[i]}_{y_true[i]}_mask_true.png"))
    

    # Optional: nifti dumps (slow; keep OFF for speed unless needed)
    if save_nifti:
        nib.save(nib.Nifti1Image(mask_pred[i].astype(np.float32), None),
                 os.path.join(nifti_dir, f"{i}_{y_pred[i]}_{y_true[i]}_mask_pred_heat.nii.gz"))
        nib.save(nib.Nifti1Image(gt.astype(np.uint8), None),
                 os.path.join(nifti_dir, f"{i}_{y_pred[i]}_{y_true[i]}_mask_true_bin.nii.gz"))

        img = image[i]
        # if image is (C,H,W) with C=4, store as (H,W,C) for nifti convenience
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] in (1, 4):
            img = np.transpose(img, (1, 2, 0))
        nib.save(nib.Nifti1Image(img.astype(np.float32), None),
                 os.path.join(nifti_dir, f"{i}_{y_pred[i]}_{y_true[i]}_image.nii.gz"))
        


# Main entry called from Lightning hook
def result_analyser(y_pred, y_true, mask_pred, mask_true, classes, path, image, offset, out_act, network, gidx=None, slices_per_patient=155):
    """
    path: output folder (e.g. .../Results/trainID)
    """

    ensure_dir(path)

    classes =  ['Tumor-free', 'LGG', 'HGG']

    # ---- Classification outputs (fast) ----
    save_confusion_matrix(y_pred, y_true, classes, os.path.join(path, "cm.png"))

    classify_rprt = classification_report(y_true, y_pred, zero_division=1)
    jindex = round(jaccard_score(y_true, y_pred, average="weighted"), 4)

    with open(os.path.join(path, "results.txt"), "a") as f:
        f.write("\nClassification report:\n")
        f.write(str(classify_rprt) + "\n")
        f.write(f"Jaccard index (weighted) = {jindex}\n")

    # ---- Segmentation metrics (Phase A: NO SAVING) ----
    if mask_pred is None:
        return

    records = []
    N = len(mask_pred)
    clean_masks = [None] * N

    pred_raw_bin = [None] * N
    pred_clean_bin = [None] * N
    gt_bin = [None] * N
    pid_arr = None
    if gidx is not None:
        pid_arr = (np.asarray(gidx).astype(int) // int(slices_per_patient))



    for i in range(N):

        hm = np.squeeze(mask_pred[i])


        raw_mask, info, _hm_sup = threshold_heatmap_to_mask(
            hm_raw=hm,
            out_act=out_act,          # <-- ADD THIS
            prefer="auto",
            fixed_thr=0.5,
            multiotsu_classes=3,
            offset=offset,
            normalize=True,
            network=network,
        )


        clean_mask = clean_binary_mask(
            raw_mask,
            min_obj_px=64,
            hole_area_px=128,
            closing_radius=2,
            keep_k=1,
        )



        gt = np.squeeze((mask_true[i] > 0).astype(np.uint8))

        pred_raw_bin[i] = raw_mask.astype(np.uint8)
        pred_clean_bin[i] = clean_mask.astype(np.uint8)
        gt_bin[i] = gt.astype(np.uint8)


        gt_sum = int(gt.sum())
        raw_sum = int(raw_mask.sum())
        clean_sum = int(clean_mask.sum())

        if gt_sum > 0:
            dice_raw, iou_raw = segscores(raw_mask.astype(np.float32), gt.astype(np.float32))
            dice_clean, iou_clean = segscores(clean_mask.astype(np.float32), gt.astype(np.float32))
            use_for_tumour_dice = True
        else:
            dice_raw = iou_raw = np.nan
            dice_clean = iou_clean = np.nan
            use_for_tumour_dice = False

        records.append({
            "slice_idx": i,
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
            "gt_sum": gt_sum,
            "pred_sum_raw": raw_sum,
            "pred_sum_clean": clean_sum,
            "dice_raw": dice_raw,
            "dice_clean": dice_clean,
            "iou_raw": iou_raw,
            "iou_clean": iou_clean,
            "use_for_tumour_dice": use_for_tumour_dice,
            "thr_method": info.get("method"),
            "thr_value": info.get("thr"),
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(path, "slice_metrics.csv"), index=False)


    # ---- Report tumour-only medians + healthy FP-seg rate ----
    tum = df[df["use_for_tumour_dice"] == True].copy()

    # RAW medians
    med_all_tumour_raw = float(tum["dice_raw"].median()) if len(tum) else float("nan")
    med_lgg_raw = float(tum[tum["y_true"] == 1]["dice_raw"].median()) if len(tum[tum["y_true"] == 1]) else float("nan")
    med_hgg_raw = float(tum[tum["y_true"] == 2]["dice_raw"].median()) if len(tum[tum["y_true"] == 2]) else float("nan")

    # CLEAN medians
    med_all_tumour_clean = float(tum["dice_clean"].median()) if len(tum) else float("nan")
    med_lgg_clean = float(tum[tum["y_true"] == 1]["dice_clean"].median()) if len(tum[tum["y_true"] == 1]) else float("nan")
    med_hgg_clean = float(tum[tum["y_true"] == 2]["dice_clean"].median()) if len(tum[tum["y_true"] == 2]) else float("nan")

    healthy = df[(df["y_true"] == 0) & (df["gt_sum"] == 0)]
    fp_rate = float((healthy["pred_sum_clean"] > 0).mean()) if len(healthy) else float("nan")

    with open(os.path.join(path, "results.txt"), "a") as f:
        f.write(f"Tumour-only Dice median (raw)  - all tumour: {med_all_tumour_raw:.4f}\n")
        f.write(f"Tumour-only Dice median (raw)  - LGG: {med_lgg_raw:.4f}\n")
        f.write(f"Tumour-only Dice median (raw)  - HGG: {med_hgg_raw:.4f}\n")
        f.write(f"Tumour-only Dice median (clean) - all tumour: {med_all_tumour_clean:.4f}\n")
        f.write(f"Tumour-only Dice median (clean) - LGG: {med_lgg_clean:.4f}\n")
        f.write(f"Tumour-only Dice median (clean) - HGG: {med_hgg_clean:.4f}\n")
        f.write(f"Healthy FP segmentation rate (clean): {fp_rate:.4f}\n")


    # ---- Phase B: Selective saving (ONLY chosen slices) ----
    selected = pick_examples(df, K_worst=30, K_best=30, K_misclf=60, K_fp_healthy=60)

    sel_dir = os.path.join(path, "Selected")
    ensure_dir(sel_dir)

    # Subfolders for nicer organization
    ensure_dir(os.path.join(sel_dir, "misclassified"))
    ensure_dir(os.path.join(sel_dir, "worst_LGG"))
    ensure_dir(os.path.join(sel_dir, "worst_HGG"))
    ensure_dir(os.path.join(sel_dir, "best_LGG"))
    ensure_dir(os.path.join(sel_dir, "best_HGG"))
    ensure_dir(os.path.join(sel_dir, "healthy_FP"))

    # Build sets for routing to folders
    mis_set = set(df[df["y_true"] != df["y_pred"]]["slice_idx"].astype(int).tolist())

    tum_only = df[df["use_for_tumour_dice"] == True].copy()
    worst_lgg_set = set(tum_only[tum_only["y_true"] == 1].nsmallest(30, "dice_clean")["slice_idx"].astype(int).tolist())
    worst_hgg_set = set(tum_only[tum_only["y_true"] == 2].nsmallest(30, "dice_clean")["slice_idx"].astype(int).tolist())
    best_lgg_set  = set(tum_only[tum_only["y_true"] == 1].nlargest(30, "dice_clean")["slice_idx"].astype(int).tolist())
    best_hgg_set  = set(tum_only[tum_only["y_true"] == 2].nlargest(30, "dice_clean")["slice_idx"].astype(int).tolist())

    healthy_fp_set = set(df[(df["y_true"] == 0) & (df["gt_sum"] == 0) & (df["pred_sum_clean"] > 0)]
                         .nlargest(60, "pred_sum_clean")["slice_idx"].astype(int).tolist())

    for i in selected:
        if i in mis_set:
            out_dir = os.path.join(sel_dir, "misclassified")
        elif i in worst_lgg_set:
            out_dir = os.path.join(sel_dir, "worst_LGG")
        elif i in worst_hgg_set:
            out_dir = os.path.join(sel_dir, "worst_HGG")
        elif i in best_lgg_set:
            out_dir = os.path.join(sel_dir, "best_LGG")
        elif i in best_hgg_set:
            out_dir = os.path.join(sel_dir, "best_HGG")
        elif i in healthy_fp_set:
            out_dir = os.path.join(sel_dir, "healthy_FP")
        else:
            out_dir = sel_dir  # fallback

        # IMPORTANT: keep save_nifti=False for speed
        save_slice_outputs(
            i=i,
            y_pred=y_pred,
            y_true=y_true,
            mask_pred=mask_pred,
            mask_true=mask_true,
            image=image,
            offset=offset,
            out_act=out_act,
            out_dir=out_dir,
            save_nifti=False,
            network=network,
            )

    # Final important line:
    print(f"[result_analyser] wrote slice_metrics.csv and saved {len(selected)} selected examples into {sel_dir}")
