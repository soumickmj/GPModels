import os
import pickle
import sys
from argparse import ArgumentParser
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torchio as tio
import torchvision.models as models
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from model.tvmodels import TVModelWrapper
from model.ReconResNet.GP_ReconResNet import GP_ReconResNet
from model.Unet.GP_UNet import GP_UNet
from model.ShuffleUnet.GP_ShuffleUNet import GP_ShuffleUNet
from model.InceptionNet import InceptionNet

from model.MProtoNet.MProtoNet2D import MProtoNet2D
from utilities.dataset import TumourDataset
from utilities.utils import (Dice, getValStat, result_analyser, save_fold_predictions)  ##TODO Save fold is new in utilities

from pathlib import Path
import pandas as pd
from SOTA_expl_comparison import (eval_explanation_map, save_overlay_flair, _binarize_multiotsu_np, _binarize_topk_np, _minmax01, _minmax01_torch, _binarize_topk_torch, dice_iou_from_masks, median_iqr)
import gc



seed_everything(1701, workers=True)

class Classifier(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self._data_prepared = False

        if self.hparams.Dataset == "China" in self.hparams.trainID:
            if len(self.classIDs) == 3:
                self.classes = ['Meningioma', 'Glioma', 'Pitutary']
            elif len(self.classIDs) == 4:
                self.classes = ['Healthy', 'Meningioma', 'Glioma', 'Pitutary']
            else:
                sys.exit("Number of classes is other than 3 or 4")

        elif self.hparams.Dataset == "Brats20":
            self.classes = ['Healthy', 'LGG', 'HGG']

        # tio_trans = {
        #             tio.RandomMotion(): 1,
        #             tio.RandomGhosting(): 1,
        #             tio.RandomBiasField(): 1,
        #             tio.RandomBlur(): 1,
        #             tio.RandomNoise(): 1,
        #         }
        # self.tio_trans = transforms.Compose([
        #                                         toTorchIO(),
        #                                         tio.Compose([tio.OneOf(tio_trans, p=0.5)]),
        #                                         fromTorchIO()
        #                                     ])

        self.trans = [ transforms.ToTensor(), ]

        if self.hparams.Dataset == "China":
            self.trans_aug = [
                                transforms.ToPILImage(),
                                transforms.RandomRotation((-330, +330)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                            ]
            self.transform1 = self.trans_aug + self.trans

        elif self.hparams.Dataset == "Brats20":
            self.trans_aug = [
                                transforms.ToTensor(),
                                transforms.RandomRotation((-330, +330)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                            ]
            self.transform1 = self.trans_aug

        # self.transform2 = [self.tio_trans] + self.trans_aug + self.trans

        if kwargs['normmode'] == 0: #Z-Score normalisation
            self.trans.append(transforms.Normalize([self.DSmean], [self.DSstd]))
        elif kwargs['normmode'] == 1: #Divide by nth percentile (default: 99th)
            self.trans.append(transforms.Normalize([0], [self.DSperval]))
        elif kwargs['normmode'] == 2: #Divide by Max
            self.trans.append(transforms.Normalize([0], [self.DSmax]))


        if kwargs['network'] == "resnet18":
            self.net = TVModelWrapper(model=models.resnet18, num_classes=len(self.classIDs))
        elif kwargs['network'] == "GP_UNet":
            if self.hparams.Relu == "leaky":
                self.net = GP_UNet(in_channels= (4 if self.hparams.contrast == "allCont" else 1), n_classes=len(self.classIDs),up_mode=self.hparams.upalgo, Relu = self.hparams.Relu)
            else:
                self.net = GP_UNet(in_channels= (4 if self.hparams.contrast == "allCont" else 1), n_classes=3,up_mode=self.hparams.upalgo, dropout=self.hparams.dropout, out_act=self.hparams.out_act)

        elif kwargs['network'] == "GP_ReconResNet":
            self.net = GP_ReconResNet(in_channels= (4 if self.hparams.contrast == "allCont" else 1), n_classes=3, res_drop_prob=self.hparams.dropout, out_act=self.hparams.out_act, upinterp_algo=self.hparams.upalgo)

        elif kwargs['network'] == "GP_ShuffleUNet":
            # self.net = GP_ShuffleUNet(in_ch= (4 if self.hparams.contrast == "allCont" else 1), out_ch=3, dropout=self.hparams.dropout, out_act=self.hparams.out_act)
            self.net = GP_ShuffleUNet(d=2, in_ch=(4 if self.hparams.contrast == "allCont" else 1), num_features=64, n_levels=3, out_ch=3, dropout=self.hparams.dropout, out_act=self.hparams.out_act) #, Relu = self.hparams.Relu

        elif kwargs['network'] == "Inception_v3":
            self.net = InceptionNet(model=models.inception_v3, in_channels= (4 if self.hparams.contrast == "allCont" else 1), num_classes=3)

        self.classWs = np.array([0.63720577, 8.99051117, 0.75790886], dtype=np.float32)  ##hardcoded from Brats20 because loading full dataset and calculating them from prepare_dataset_splits(self) is no longer called twice for faster flow, TODO Calculate them once on your dataset and hardcode it
        self.loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.classWs).float() if kwargs['useClassWeight'] else None)
        self.mask_accuracy = Dice()


        # ---- Lightning v2 epoch buffers ----
        self._train_losses = []
        self._val_losses = []
        self._val_correct = 0
        self._val_count = 0
        self._val_dices = []

        self._test_losses = []
        self._test_preds = []
        self._test_labels = []
        self._test_probs = []
        self._test_pred_masks = []
        self._test_masks = []
        self._test_images = []

        # ---- Explainability eval buffers (Top-k / Multi-Otsu) ----
        self._test_attr_maps = []   # [B,H,W] continuous attribution (single-channel)
        self._test_gt_wt = []       # [B,H,W] ground truth whole-tumor mask (binary)
        self._test_y = []           # [B] labels for per-class reporting (optional)

        self._vis_saved = 0
        self._vis_max_per_fold = 10

        # # ---- explainability buffers (init once) ----
        self._test_expl_maps = []     # stores continuous maps [B,H,W] or [B,1,H,W]






    def setup(self, stage: str | None = None):
        # load data only once, and only when actually needed
        if self._data_prepared:
            return

        if stage in (None, "fit", "validate", "test", "predict"):
            self.prepare_dataset_splits()
            self._data_prepared = True


    def mask_merge(self, mask, pred=None):
        if self.hparams.MaskMergeMode == "sum":
            return torch.sum(mask, dim=1, keepdims=True)
        else:
            selected_mask = []
            pred_hat = torch.argmax(pred, dim=1)
            for i in range(len(mask)):
                mask = mask.float() #added this to solve utils line 52 unsupported dtype when using predselect mode
                selected_mask.append(mask[i, pred_hat[i], ...])
            return torch.stack(selected_mask).unsqueeze(1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.w_d
        )
        return {
            'optimizer': optimizer,
            'monitor': 'val_loss',
        }


    def forward(self, x):
        # Ensure NCHW for torchvision ResNet and our MProtoNet2D
        # If x is [B, H, C, W], convert to [B, C, H, W]
        if x.dim() == 4 and x.shape[1] != (4 if self.hparams.contrast == "allCont" else 1) and x.shape[2] in (1, 3, 4):
            x = x.permute(0, 2, 1, 3).contiguous()

        return self.net(x)


    def training_step(self, batch, batch_idx):
        image, _, label = batch
        out = self(image)
        loss = self.loss(out, label)

        if loss != loss:
            print("loss is nan")

        # log per-step + epoch
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=image.size(0))

        # store for custom epoch reduction (median)
        self._train_losses.append(loss.detach())

        return loss



    def validation_step(self, batch, batch_idx):
        image, mask, label = batch

        if self.hparams.model_segclassify:
            out, pred_mask = self(image)
            pred_mask = self.mask_merge(pred_mask, out)
            # pred_mask = (pred_mask > 0.5).float()
            _, dice = self.mask_accuracy(pred_mask, mask)
        else:
            out, dice = self(image), None

        loss = self.loss(out, label)
        label_hat = torch.argmax(out, dim=1)
        n_correct = (label == label_hat).sum().item()

        # store for epoch-end aggregation
        self._val_losses.append(loss.detach())
        self._val_correct += int(n_correct)
        self._val_count += int(label.numel())
        if dice is not None:
            self._val_dices.append(dice.detach())

        # optional logging
        self.log("val_loss_step", loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=image.size(0))
        return loss

    def on_test_start(self):
        self._vis_saved = 0
        self._test_expl_maps = []





    #### new Code.... TODO
    def test_step(self, batch, _):

        image, mask, label= batch

        # --- FIX GT ORIENTATION ---
        mask = torch.rot90(mask, k=-1, dims=(-2, -1))  # rotate 90° right
        mask = torch.flip(mask, dims=(-1,))            # flip left-right

        if self.hparams.model_segclassify:
            logits, pred_mask = self(image)  # pred_mask is likely [B,C,H,W] (continuous)

            probs = torch.softmax(logits, dim=1)
            pred_cls = torch.argmax(probs, dim=1)  # [B]

            if self.hparams.MaskMergeMode == "predselect":
                pred_mask_merged = self.mask_merge(pred_mask, logits)  # [B,1,H,W]
            else:
                expl_map = []
                for i in range(pred_mask.shape[0]):
                    expl_map.append(pred_mask[i, pred_cls[i], ...])
                expl_map = torch.stack(expl_map, dim=0)  # [B,H,W]
                # cache expl map + GT mask
                self._test_expl_maps.append(expl_map.detach().cpu())


            pred_mask_merged = self.mask_merge(pred_mask, logits).detach().cpu()
            pred_mask_cpu = pred_mask_merged
        else:
            logits, pred_mask = self(image), None
            probs = torch.softmax(logits, dim=1)


        probs = torch.softmax(logits, dim=1)
        loss = self.loss(logits, label)
        label_hat = torch.argmax(probs, dim=1)


        # cache for epoch-end analysis
        self._test_losses.append(loss.detach().cpu())
        self._test_preds.append(label_hat.detach().cpu())
        self._test_labels.append(label.detach().cpu())
        self._test_probs.append(probs.detach().cpu())
        self._test_images.append(image.detach().cpu())

        # always cache GT masks (needed for patient aggregation, explainabilty, dice)
        self._test_masks.append(mask.detach().cpu())

        # cache predicted masks only if available
        if self.hparams.model_segclassify:
            self._test_pred_masks.append(pred_mask_cpu)

        return loss


    def prepare_dataset_splits(self):

        ###Saved Testing Pickles for inference fold 0
        PICKLE_DIR = Path(f"/project/Data/GPModels/Dataset/PreProcessed_Brats2020/pickles/full/Axi")

        with open(PICKLE_DIR / "X_allNonEmptySlices.pickle", "rb") as f:
            X = pickle.load(f)

        with open(PICKLE_DIR / "m_allNonEmptySlices.pickle", "rb") as f:
            m = pickle.load(f)

        with open(PICKLE_DIR / "y_allNonEmptySlices.pickle", "rb") as f:
            y = pickle.load(f)


        print("Loaded pickles:")
        print("X shape:", X.shape)  # (N, 4, H, W)
        print("m shape:", m.shape)  # (N, 1, H, W)
        print("y shape:", y.shape)  # (N,)
        print("Classes present:", np.unique(y))

        Debug = False
        if Debug:
            DEBUG_SAMPLES_PER_CLASS = 64

            indices = []
            for cls in np.unique(y):
                cls_idx = np.where(y == cls)[0][:DEBUG_SAMPLES_PER_CLASS]
                indices.append(cls_idx)

            indices = np.concatenate(indices)

            X = X[indices]
            m = m[indices]
            y = y[indices]

            print("DEBUG shapes:")
            print(X.shape, m.shape, y.shape)
            print("Class distribution:", {c: np.sum(y == c) for c in np.unique(y)})

        y = y.astype(np.int64)
        if y.min() == 1: y-=1 #The initial class should be 0, not 1
        self.classIDs = np.unique(y)

        ind_trainval, ind_test  = list(StratifiedShuffleSplit(n_splits=1, test_size=self.hparams.test_percent, random_state=13).split(X, y))[0]
        self.X_test = X[ind_test]
        self.y_test = y[ind_test]
        self.m_test = m[ind_test]
        X = X[ind_trainval]
        y = y[ind_trainval]
        m = m[ind_trainval]

        self.classWs = compute_class_weight(class_weight='balanced',classes=self.classIDs,y=y)
        self.class_weight = torch.tensor(self.classWs, dtype=torch.float32)
        self.DSmean, self.DSstd, self.DSperval, self.DSmax, self.DSmin = X.mean(), X.std(), np.percentile(X, self.hparams.percentile), X.max(), X.min()

        ind_train, ind_val  = list(StratifiedShuffleSplit(n_splits=self.hparams.n_folds, test_size=self.hparams.val_percent, random_state=42).split(X, y))[self.hparams.foldID]
        self.X_train = X[ind_train]
        self.y_train = y[ind_train]
        self.m_train = m[ind_train]
        self.X_val = X[ind_val]
        self.y_val = y[ind_val]
        self.m_val = m[ind_val]


    def train_dataloader(self) -> DataLoader:
        return DataLoader(TumourDataset(self.hparams.contrast ,self.X_train, self.m_train, self.y_train, maxinorm=True if self.hparams.normmode == 3 else False,
                          transforms=transforms.Compose(self.transform2 if self.hparams.MRIAug else self.transform1), trans=transforms.Compose(self.trans)),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers)    #, collate_fn=collate_fn

    def val_dataloader(self) -> DataLoader:
        return DataLoader(TumourDataset(self.hparams.contrast , self.X_val, self.m_val, self.y_val, maxinorm=True if self.hparams.normmode == 3 else False, transforms=transforms.Compose(self.trans), trans=transforms.Compose(self.trans)),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers) #, collate_fn=collate_fn

    def test_dataloader(self) -> DataLoader:
        return DataLoader(TumourDataset(
            self.hparams.contrast, self.X_test, self.m_test, self.y_test,
            maxinorm=True if self.hparams.normmode == 3 else False,
            transforms=transforms.Compose(self.trans),
            trans=transforms.Compose(self.trans),
        ), shuffle=False, batch_size=self.hparams.batch_size,
                        pin_memory=True, num_workers=self.hparams.workers)

    def on_train_epoch_end(self):
        if len(self._train_losses) > 0:
            avg_loss = torch.stack(self._train_losses).median()
            self.log("training_loss", avg_loss, prog_bar=True)
        self._train_losses.clear()


    def on_validation_epoch_end(self):
        if len(self._val_losses) > 0:
            avg_loss = torch.stack(self._val_losses).median()
            self.log("val_loss", avg_loss, prog_bar=True)

        if self._val_count > 0:
            val_acc = self._val_correct / float(self._val_count)
            self.log("accuracy", val_acc, prog_bar=True)

        if self.hparams.model_segclassify and len(self._val_dices) > 0:
            dice_med = torch.stack(self._val_dices).median()
            self.log("dice", dice_med, prog_bar=True)

        # clear buffers
        self._val_losses.clear()
        self._val_dices.clear()
        self._val_correct = 0
        self._val_count = 0

    def on_test_epoch_end(self):

        preds = torch.cat(self._test_preds) if len(self._test_preds) else torch.tensor([])
        labels = torch.cat(self._test_labels) if len(self._test_labels) else torch.tensor([])
        probs = torch.cat(self._test_probs) if len(self._test_probs) else torch.tensor([])
        images = torch.cat(self._test_images) if len(self._test_images) else torch.tensor([])

        if len(self._test_losses) > 0:
            test_loss = torch.stack(self._test_losses).median()
            self.log('test_loss', test_loss)

        if len(labels) > 0:
            test_acc = torch.sum(labels == preds).item() / float(len(labels))
            self.log('test_acc', test_acc)

        if self.hparams.model_segclassify and len(self._test_masks) > 0:
            mask = torch.cat(self._test_masks).squeeze().numpy()
            pred_mask = torch.cat(self._test_pred_masks).squeeze().numpy()
        else:
            mask, pred_mask = None, None

        image = images.squeeze().numpy() if len(images) > 0 else None

        offset = 0

        result_analyser(
            preds,
            labels,
            pred_mask,
            mask,
            self.classes,
            os.path.join(self.hparams.out_path, "Results", self.hparams.trainID),
            image,
            offset,
            self.hparams.out_act,
            self.hparams.network,
        )


        ####TODO NEW: save fold predictions
        probs_np = probs.numpy()
        labels_np = labels.numpy()
        out_dir = os.path.join(self.hparams.out_path, "Results", self.hparams.trainID)
        fold_id = int(self.hparams.foldID) if hasattr(self.hparams, "foldID") else 0
        save_fold_predictions(labels_np, probs_np, out_dir, fold_id)

        # clear buffers
        self._test_expl_maps.clear()





    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--main_path', type=str)
        parser.add_argument('--out_path', type=str)
        parser.add_argument('--out_act', type=str)
        parser.add_argument('--Dataset', type=str)
        parser.add_argument('--contrast', type=str)
        parser.add_argument('--orient', type=str)
        parser.add_argument('--amount', type=str)
        parser.add_argument('--network', type=str)
        parser.add_argument('--normmode', type=int)
        parser.add_argument('--n_folds', type=int)
        parser.add_argument('--foldID', type=int)
        parser.add_argument('--model_segclassify', type=bool)
        parser.add_argument('--useClassWeight', type=bool)
        parser.add_argument('--MRIAug', type=bool)
        parser.add_argument('--MaskMergeMode', type=str)
        parser.add_argument('--test_percent', type=float)
        parser.add_argument('--val_percent', type=float)
        parser.add_argument('--trainID', type=str)
        parser.add_argument('--upalgo', type=str)
        parser.add_argument('--dropout', type=bool)
        parser.add_argument('--Relu', type=str)
        parser.add_argument('--depth', type=int)
        parser.add_argument('--wf', type=int)
        parser.add_argument('--threshmode', type=str)
        return parser

def main():
    torch.set_num_threads(1)
    resume_from_checkpoint = True
    chkp_type = "Last"  #"Best"  #"Last"
    wnbactive = False
    network: str = "GP_ShuffleUNet"  #"GP_UNet" #"resnet18" # "GP_ReconResNet", GP_ShuffleUNet,Inception_v3, MProtoNet2D
    out_act: str = "None"  #softmax or None or "sigmoid" or "relu"
    #run_prefix="InitialGPUNet_150epoch_wnbactive_deterministic"
    # run_prefix="GPUNet_150e_wnbact_deter"    #   _2  _bilinear _sinc
    num_epochs: int = 300     #150
    run_prefix=f"{network}_{num_epochs}e"    # _wnbact_deter  _2  _bilinear _sinc
    use_amp: bool = True     #Automatic Mixed Precission
    lr:float = 1e-3    #found by findlr on pc 4.786300923226383e-07    #Default: 1e-3
    batch_size:int = 8      #for Seg batch size 8 and less succeded acording to autoScaleBatchSize
    accumulate_grad_batches: int = 16        #248*249 = 32 or 64 for all models and orientations
    workers: int = 2        #stimulate 4 #GPU18 8
    normmode: int = 3 #0: ZNorm, 1: Divide by nth percentile, 2: Divide by Max
    percentile: int = 99 #nth percentile to be used only for normmode 1
    depth: int = 3    #for GP_UNet depth: default = 3, can go deeper or shallower #5 is the depth of original GPUNet
    wf: int = 6       #for GP_UNet no of filters in each layer = 2**wf  default: 6  meaning starting 64 then 128, 256 till depth 3
    model_segclassify: bool = False if network == "Inception_v3" else True         # when model is GP- then true
    useClassWeight: bool = True
    main_path: str = '/project/Data/GPModels'
    out_path: str = '/project/Data/GPModels/outputs'
    Dataset: str = "Brats20"     #China or Brats20
    dropout: bool = True if Dataset == "Brats20" else False   # default=False, or True which mean a p of 0.5 as default cause it is not specified in gpunet model
    w_d:float = 5e-4 if Dataset == "Brats20" else 0     # 5e-4 proposed for CNN and 0 for no regulirization
    g_c_val: int = 1 if Dataset == "Brats20" else 0
    g_c_algo: str = 'value'   #'norm' by default, or set to "value"
    contrast: str = "allCont"     #t1, t1ce, t2, flair, allCont
    orient: str = "Axi"     #All Sag Cor Axi
    amount: str = "full"     #amount taken from complete dataset f
    test_percent: float = 0.25
    val_percent: float = 0.20
    n_folds: int = 5
    foldID: int = 0
    MRIAug: bool = False
    upalgo: str = "sinc"  #alwas sinc from now on GPUNet "upconv" =convtrans, "bilinear" = conv + interpolate(bilinear) "sinc" = conv + interpolate(bilinear)  #GPRecon "convtrans" = transconv, "bilinear", "sinc" = two diff interpolate algo for upsampling (conve + interpolate)
    Relu: str = "Relu" #leaky or Relu
    MaskMergeMode: str = "predselect" #sum or predselect as saved in checkpoint
    threshmode: str = "try_all" #
    autoScaleBatchSize = False
    findLR = False
    runMode = "onlyTest"    # "onlyTrain" # "onlyTest" # "TrainTest"
    changeAnyTesthparam = False  #True when wanting to change any Test hparams in a saved checkpiont when resuming TrainTest or onlyTest

    if MaskMergeMode == "sum":
        ChngdMaskMergeMode = "predselect"
    elif MaskMergeMode == "predselect":
        ChngdMaskMergeMode = "sum"

    if Relu == "leaky":
        trainID = run_prefix + "_" + Dataset + "_" + amount + "_" + contrast + "_" + orient + "_"  + network + "_" + upalgo + "_" + Relu + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
                "_" + ("WMRIAug" if MRIAug else "WoMRIAug") + ("_Mask"+MaskMergeMode if model_segclassify else "") +\
                "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)

    elif "Initial" in run_prefix:
        trainID = run_prefix + "_" + orient + "_"  + network + "_" + upalgo + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
        "_" + ("WMRIAug" if MRIAug else "WoMRIAug") + ("_Mask"+MaskMergeMode if model_segclassify else "") +\
        "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)

    elif Dataset == "China":
        trainID = run_prefix + "_" + Dataset + "_" + amount + "_" + contrast + "_" + orient + "_"  + network + "_" + upalgo + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
                "_" + ("WMRIAug" if MRIAug else "WoMRIAug") + ("_Mask"+MaskMergeMode if model_segclassify else "") +\
                "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)
        #+ "_" + Dataset + "_" + amount
    else:       #Brats2020
        trainID = run_prefix + "_" + Dataset + "_" + amount + "_" + contrast + "_" + orient + "_" + network + "_"  + str(g_c_val) + "-" + str(g_c_algo) + "_w_d-" + str(w_d) + "_drop-" + str(dropout) + "_" + upalgo + \
                "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)

    ##+ "_" + upalgo
    ##+ "_" + ("WClsW" if useClassWeight else "WoClsW")
    # "_" + ("WMRIAug" if MRIAug else "WoMRIAug") +
    #+ "_fold" + str(foldID) + "of" + str(n_folds)
    ##+ "_depth" + str(depth) ##+ "_" + upalgo

    parser = ArgumentParser()
    parser = Classifier.add_model_specific_args(parser)
    # parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.out_act = out_act ####
    hparams.lr = lr
    hparams.w_d = w_d
    hparams.g_c_val = g_c_val
    hparams.g_c_algo = g_c_algo
    hparams.max_epochs = num_epochs
    hparams.batch_size = batch_size
    hparams.trainID = trainID
    hparams.normmode = normmode
    hparams.percentile = percentile
    hparams.network = network
    hparams.model_segclassify = model_segclassify
    hparams.accumulate_grad_batches = accumulate_grad_batches
    hparams.use_amp = use_amp
    hparams.workers = workers
    hparams.useClassWeight = useClassWeight
    hparams.main_path = main_path
    hparams.out_path = out_path
    hparams.Dataset = Dataset
    hparams.contrast = contrast
    hparams.orient = orient
    hparams.amount = amount
    hparams.test_percent = test_percent
    hparams.val_percent = val_percent
    hparams.n_folds = n_folds
    hparams.foldID = foldID
    hparams.MRIAug = MRIAug
    hparams.MaskMergeMode = MaskMergeMode
    hparams.autoScaleBatchSize = autoScaleBatchSize
    hparams.findLR = findLR
    hparams.upalgo = upalgo
    hparams.dropout = dropout
    hparams.Relu = Relu
    hparams.depth = depth
    hparams.wf = wf
    hparams.threshmode = threshmode

    if not wnbactive:
        os.environ["WANDB_MODE"] = "dryrun"


    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(hparams.out_path, "Checkpoints", hparams.trainID),
        monitor='val_loss',
        save_last=True,
    )
    logger = WandbLogger(name=trainID, id=trainID, project='MasterThesis_Hadya',
                            group='Baseline', entity='mickchimp', config=hparams)

    #specify which checkpoint
    if resume_from_checkpoint:
        if chkp_type == "Best":
            checkpoint_dir = pjoin(hparams.out_path, "Checkpoints", hparams.trainID)
            chkpoint = pjoin(checkpoint_dir, sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1])
        elif chkp_type == "Last":
            # chkpoint = pjoin(hparams.out_path, "Checkpoints", hparams.trainID, "last.ckpt")
            if network == "GP_UNet":
                chkpoint = "/project/Data/GPModels/outputs/checkpoints/fold0/GPUNet/epoch=291-step=65407.ckpt"
                # chkpoint = "/project/Data/GPModels/outputs/checkpoints/fold1/GPUNet/epoch=292-step=65631.ckpt"
            elif network == "GP_ReconResNet":
                chkpoint = "/project/Data/GPModels/outputs/checkpoints/fold0/GPRecon/epoch=274-step=61599.ckpt"
            elif network == "GP_ShuffleUNet":
                chkpoint = "/project/Data/GPModels/outputs/checkpoints/fold0/GPShuffleUNet/epoch=276-step=62047.ckpt"
            elif network == "Inception_v3":
                chkpoint = "/project/Data/GPModels/outputs/checkpoints/fold0/Inceptionv3/epoch=271-step=60927.ckpt"
            elif network == "MProtoNet2D":
                chkpoint = "/project/Data/GPModels/outputs/Checkpoints/Org_MProtoNet2D_300e_Brats20_full_allCont_Axi_MProtoNet2D_1-value_w_d-0.0005_drop-True_sinc_norm3_fold0of5/epoch=274-step=202125.ckpt"



    else:

        chkpoint = None

    if not changeAnyTesthparam:
        os.makedirs(pjoin(hparams.out_path, "Results", hparams.trainID), exist_ok=True)


    use_gpu = torch.cuda.is_available()

    # train
    trainer = Trainer(
        logger=logger,
        # precision="16-mixed" if (use_gpu and use_amp) else 32,  # avoid precision=16 warning
        precision= 32,  # avoid precision=16 warning
        # gpus=1,     #gpus=1   ##deprecated
        accelerator="gpu" if use_gpu else "cpu",  ##instead of gpus=1
        devices=1 if use_gpu else "auto",          ##instead of gpus=1
        # checkpoint_callback=True,  #checkpoint_callback=checkpoint_callback,   was for old trainer.py istead of 326, 327 # remove this, use callbacks list instead:
        callbacks=[checkpoint_callback],
        max_epochs=hparams.max_epochs,
        # terminate_on_nan=True,        use  detect_anomaly=True,  instead
        deterministic=True,
        accumulate_grad_batches=accumulate_grad_batches,
        # resume_from_checkpoint=chkpoint,        # deprecated: handle when calling trainer.fit(...)
        # auto_scale_batch_size='binsearch' if autoScaleBatchSize else None,
        # auto_lr_find=findLR,
        gradient_clip_val=g_c_val,
        gradient_clip_algorithm=g_c_algo
    )

    # Only train
    if runMode == "onlyTrain":
        model = Classifier(**vars(hparams))
        logger.watch(model, log='all', log_freq=100)

        if autoScaleBatchSize or findLR:
            trainer.tune(model)

        if chkpoint is not None:
            trainer.fit(model, ckpt_path=chkpoint)
        else:
            trainer.fit(model)


    # Only test using the best model!
    elif runMode == "onlyTest":
        use_amp = False # disable AMP for testing to avoid potential issues
        if chkpoint is None:
            raise RuntimeError("No checkpoint provided")

        torch.cuda.empty_cache()
        gc.collect()

        ckpt = torch.load(chkpoint, map_location="cpu", weights_only=False)

        ckpt_hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams")
        if ckpt_hparams is None:
            raise KeyError("Checkpoint missing hyper_parameters/hparams")

        if not isinstance(ckpt_hparams, dict):
            ckpt_hparams = vars(ckpt_hparams)

        # (optional) overwrite paths if they differ on this machine
        ckpt_hparams["main_path"] = main_path
        ckpt_hparams["out_path"] = out_path

        model = Classifier(**ckpt_hparams)

        sd = ckpt["state_dict"]
        model.load_state_dict(sd, strict=True)

        trainer.test(model=model, ckpt_path=None)



    #else training and testing (resuming training or from scratch)
    elif runMode == "TrainTest":
        model = Classifier(**vars(hparams))
        logger.watch(model, log='all', log_freq=100)

        if autoScaleBatchSize or findLR:
            trainer.tune(model)

        #1-Resume training with changing Testing hparam
        if changeAnyTesthparam:
            # resume or start from scratch depending on chkpoint
            if chkpoint is not None:
                trainer.fit(model, ckpt_path=chkpoint)
            else:
                trainer.fit(model)

            if chkpoint is None:
                raise RuntimeError("the next line will produce an error, resume_from_Checkpoint is set to False")
            model = Classifier.load_from_checkpoint(chkpoint)
            # 1- change threshold
            if threshmode:
                model.hparams.trainID = run_prefix + "_" + Dataset + "_" + amount + "_" + orient + "_"  + network + "_" + upalgo + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
                        "_" + ("WMRIAug" if MRIAug else "WoMRIAug") + ("_Mask"+MaskMergeMode if model_segclassify else "") + "_Thresh" + threshmode +\
                        "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)
            else: # 2- change mask merge mode
                model.hparams.MaskMergeMode = ChngdMaskMergeMode
                model.hparams.trainID = run_prefix + "_" + Dataset + "_" + amount + "_" + orient + "_"  + network + "_" + upalgo + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
                        "_" + ("WMRIAug" if MRIAug else "WoMRIAug") + ("_Mask_chng_"+model.hparams.MaskMergeMode if model_segclassify else "") +\
                        "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)

            #Test
            # trainer.test(model, test_dataloaders=model.test_dataloader())
            trainer.test(model=Classifier(**vars(hparams)), ckpt_path=chkpoint, weights_only=False)



        #2- resume or Train and Test from scratch without changing Testhparam
        else:  # not changeAnyTesthparam
            if chkpoint is not None:
                trainer.fit(model, ckpt_path=chkpoint)
            else:
                trainer.fit(model)

            trainer.test(test_dataloaders=model.test_dataloader())

if __name__ == '__main__':
    main()
