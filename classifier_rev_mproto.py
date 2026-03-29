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

from model.tvmodels import TVModelWrapper
from model.ReconResNet.GP_ReconResNet import GP_ReconResNet
from model.Unet.GP_UNet import GP_UNet
from model.MProtoNet.MProtoNet2D import MProtoNet2D
from utilities.dataset import TumourDataset
from utilities.utils import (Dice, result_analyser, save_fold_predictions)  ##TODO Save fold is new in utilities
from utilities.dataset import WithGlobalIndex

from pathlib import Path

from model.MProtoNet.push_2D import push_prototypes_2d
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd

import numpy as np

from utilities.interpretability import attribute_gradcam_2d, attribute_mprotonet_2d, lc_2d
from SOTA_expl_comparison import (
    eval_explanation_map, save_overlay_flair,
    _binarize_multiotsu_np, _binarize_topk_np, _minmax01,
    median_iqr
)



class NumpyCHWToTensor:
    """Convert a numpy array shaped (C,H,W) to torch.FloatTensor (C,H,W)."""
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)   # keeps CHW ordering
        if not torch.is_tensor(x):
            raise TypeError(f"Expected numpy or torch tensor, got {type(x)}")
        return x.float()



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def to_uint8_01(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] then convert to uint8 [0,255]."""
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    x = (x - mn) / (mx - mn + 1e-8)
    return (x * 255.0).clip(0, 255).astype(np.uint8)


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
        # #         }
        # self.tio_trans = transforms.Compose([
        #                                         toTorchIO(),
        #                                         tio.Compose([tio.OneOf(tio_trans, p=0.5)]),
        #                                         fromTorchIO()
        #                                     ])

        self.trans = []  # for Brats20; images already tensors


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
                NumpyCHWToTensor(),  # <-- replaces transforms.ToTensor()
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
            self.net = GP_ReconResNet(n_classes=len(self.classIDs), upinterp_algo=self.hparams.upalgo)

        elif kwargs['network'] == "MProtoNet2D":
            in_ch = (4 if self.hparams.contrast == "allCont" else 1)
            self.net = MProtoNet2D(
                in_size=(in_ch, 240, 240),
                out_size=3,
                features="resnet50_ri",              # repo default style
                n_layers=7,
                prototype_shape=(30, 128, 1, 1),
                f_dist="cos",                       # repo recommended in README
                prototype_activation_function="log",
                p_mode=5,                           # closest to “MProtoNet C”
                topk_p=1,
                init_weights=True,
            )

        self.classWs = np.array([0.63720577, 8.99051117, 0.75790886], dtype=np.float32)   ##hardcoded from Brats20, TODO Remove
        self.loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.classWs).float() if kwargs['useClassWeight'] else None)
        self.mask_accuracy = Dice()

        if kwargs['network'] == "MProtoNet2D":
            self.automatic_optimization = False
            self.stage = "joint"
            self.coefs = {
                'cls': 1,
                'clst': 0.8,
                'sep': -0.08,
                'L1': 0.01,
                'map': 0.5,
                'OC': 0.05,
            }

        # Lightning v2 epoch buffers
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

        # Explainability eval buffers (Top-k / Multi-Otsu)
        self._test_attr_maps = []   # [B,H,W] continuous attribution (single-channel)
        self._test_gt_wt = []       # [B,H,W] ground truth whole-tumor mask (binary)
        self._test_y = []           # [B] labels for per-class reporting (optional)

        self._vis_saved = 0
        self._vis_max_per_fold = 100



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
        if self.hparams.network != "MProtoNet2D":
            opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.w_d)
            return opt

        net = self.net

        params = [
            {'params': net.features.parameters(), 'lr': self.hparams.lr, 'weight_decay': self.hparams.w_d},
            {'params': net.add_ons.parameters(), 'lr': self.hparams.lr, 'weight_decay': self.hparams.w_d},
            {'params': net.prototype_vectors,     'lr': self.hparams.lr, 'weight_decay': 0.0},
        ]
        if net.p_mode >= 2:
            params += [
                {'params': net.p_map.parameters(), 'lr': self.hparams.lr, 'weight_decay': self.hparams.w_d},
            ]

        params_last = [
            {'params': net.last_layer.parameters(), 'lr': self.hparams.lr, 'weight_decay': 0.0},
        ]

        opt_joint = torch.optim.AdamW(params)      # repo default is often AdamW
        opt_last  = torch.optim.AdamW(params_last)
        return [opt_joint, opt_last]


    def forward(self, x):
        # Ensure NCHW for MProtoNet2D
        # If x is [B, H, C, W], convert to [B, C, H, W]
        if x.dim() == 4 and x.shape[1] != (4 if self.hparams.contrast == "allCont" else 1) and x.shape[2] in (1, 3, 4):
            x = x.permute(0, 2, 1, 3).contiguous()

        return self.net(x)


    def train_last_layer_loops(self, n_loops: int = 10):
        """
        From MProtoNet Repo 'last layer' optimization after pushing.
        Runs n_loops passes over the training dataloader, updating ONLY last_layer optimizer.
        """
        assert self.hparams.network == "MProtoNet2D"
        self.stage = "last"
        self.net.train()

        opt_joint, opt_last = self.optimizers()  # you already return 2 optimizers
        # only use opt_last here
        for _ in range(n_loops):
            for batch_idx, batch in enumerate(self.train_dataloader()):
                image, _, label = batch
                image = image.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)

                # forward like repo
                if self.net.p_mode >= 2:
                    logits, min_distances, x_feat, p_map = self.net(image)
                else:
                    logits, min_distances = self.net(image)

                cls_loss = self.loss(logits, label)

                l1_mask = 1 - self.net.prototype_class_identity.mT
                l1 = torch.linalg.vector_norm(self.net.last_layer.weight * l1_mask, ord=1)

                loss = self.coefs['cls'] * cls_loss + self.coefs['L1'] * l1

                opt_last.zero_grad()
                self.manual_backward(loss)
                opt_last.step()

        self.stage = "joint"


    def training_step(self, batch, batch_idx):
        image, _, label = batch
        target = label  # already class indices in your dataset

        opt_joint, opt_last = self.optimizers()
        net = self.net
        coefs = self.coefs

        # forward 
        if net.p_mode >= 2:
            output, min_distances, x, p_map = net(image)
        else:
            output, min_distances = net(image)

        # classification 
        classification = self.loss(output, target)
        loss_cls = coefs['cls'] * classification

        if self.stage in ["warm_up", "joint"]:
            # cluster / separation
            max_dist = torch.prod(torch.tensor(net.prototype_shape[1:], device=self.device)).to(self.device)

            target_weight = self.class_weight.to(self.device)[target]
            target_weight = target_weight / target_weight.sum()

            prototypes_correct = net.prototype_class_identity[:, target].mT  # [B,P]

            inv_distances_correct = ((max_dist - min_distances) * prototypes_correct).amax(1)
            cluster = ((max_dist - inv_distances_correct) * target_weight).sum()
            loss_clst = coefs['clst'] * cluster

            prototypes_wrong = 1 - prototypes_correct
            inv_distances_wrong = ((max_dist - min_distances) * prototypes_wrong).amax(1)
            separation = ((max_dist - inv_distances_wrong) * target_weight).sum()
            loss_sep = coefs['sep'] * separation

            loss = loss_cls + loss_clst + loss_sep

            # mapping loss
            if net.p_mode >= 2:
                ri = torch.randint(2, (1,), device=self.device).item()
                scale = (0.75, 0.875)[ri]
                f_affine = lambda t: F.interpolate(t, scale_factor=scale, mode='bilinear', align_corners=True)
                f_l1 = lambda t: t.abs().mean()
                mapping = f_l1(f_affine(p_map) - net.get_p_map(f_affine(x))) + f_l1(p_map)
                loss_map = coefs['map'] * mapping
                loss = loss + loss_map

            # online-CAM loss
            if net.p_mode >= 4:
                p_x = net.lse_pooling(net.p_map[:-3](x).flatten(2))
                output2 = net.last_layer(p_x @ net.p_map[-3].weight.flatten(1).mT)
                online_cam = self.loss(output2, target)
                loss_oc = coefs['OC'] * online_cam
                loss = loss + loss_oc

            opt = opt_joint

        else:
            # L1 last-layer stage
            l1_mask = 1 - net.prototype_class_identity.mT
            l1 = torch.linalg.vector_norm(net.last_layer.weight * l1_mask, ord=1)
            loss_l1 = coefs['L1'] * l1
            loss = loss_cls + loss_l1

            opt = opt_last

        # manual optimization (like mproto repo)
        opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        opt.step()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, _):
        image, mask, label = batch

        if self.hparams.model_segclassify:
            out, pred_mask = self(image)
            pred_mask = self.mask_merge(pred_mask, out)

            # IMPORTANT: threshold before Dice (recommended for both GP-UNet and MProtoNet)
            pred_mask = (pred_mask > 0.5).float()

            _, dice = self.mask_accuracy(pred_mask, mask)
            self._val_dices.append(dice.detach())
        else:
            out, pred_mask, dice = self(image), None, None

        loss = self.loss(out, label)
        self._val_losses.append(loss.detach())

        label_hat = torch.argmax(out, dim=1)
        self._val_correct += torch.sum(label == label_hat).item()
        self._val_count += label.numel()

        self.log("val_loss_step", loss, on_step=True, on_epoch=False)


    @torch.no_grad()
    def _log_proto_selectivity(self, loader, max_batches=5, prefix="val"):
        if self.hparams.network != "MProtoNet2D":
            return

        self.net.eval()

        acts_list = []
        labels_list = []

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            image, _, label = batch
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)

            # ensure class indices (not one-hot)
            if label.ndim > 1:
                label = label.argmax(1)

            if self.net.p_mode >= 2:
                logits, min_distances, *_ = self.net(image, return_min_distances=True)
            else:
                logits, min_distances = self.net(image, return_min_distances=True)

            acts = self.net.distance_2_similarity(min_distances)  # [B,P]
            acts_list.append(acts.detach().cpu())
            labels_list.append(label.detach().cpu())

        if not acts_list:
            return

        acts = torch.cat(acts_list)       # [N,P]
        labels = torch.cat(labels_list)   # [N]

        proto_class = self.net.prototype_class_identity.argmax(1)  # [P]
        P = acts.shape[1]

        own_minus_other = []
        for k in range(P):
            c = int(proto_class[k].item())
            own = acts[labels == c, k].mean() if (labels == c).any() else torch.tensor(0.)
            other = acts[labels != c, k].mean() if (labels != c).any() else torch.tensor(0.)
            own_minus_other.append((own - other).item())

        score = float(sum(own_minus_other) / len(own_minus_other))

        # logs to W&B per epoch
        self.log(f"{prefix}/proto_selectivity_own_minus_other", score, prog_bar=True, on_epoch=True)



    def on_test_start(self):
        self._lc_ap_vals = []
        self._lc_dsc_vals = []
        self._test_attr_maps = []
        self._test_gt_wt = []
        self._test_y = []
        self._vis_saved = 0


    def test_step(self, batch, _):


        image, mask, label, gidx = batch

        if self.hparams.model_segclassify:
            logits, pred_mask = self(image)
            pred_mask = self.mask_merge(pred_mask, logits)

            # threshold for Dice + for saving binary masks consistently
            pred_mask = (pred_mask > 0.5).float()

            pred_mask_cpu = pred_mask.detach().cpu()
        else:
            logits, pred_mask_cpu = self(image), None

        probs = torch.softmax(logits, dim=1)
        loss = self.loss(logits, label)
        label_hat = torch.argmax(probs, dim=1)

        # cache for epoch-end analysis
        self._test_losses.append(loss.detach().cpu())
        self._test_preds.append(label_hat.detach().cpu())
        self._test_labels.append(label.detach().cpu())
        self._test_probs.append(probs.detach().cpu())
        self._test_images.append(image.detach().cpu())

        # always cache GT masks (needed for patient aggregation)
        self._test_masks.append(mask.detach().cpu())

        # cache predicted masks only if available
        if self.hparams.model_segclassify:
            self._test_pred_masks.append(pred_mask_cpu)

        # Pred Class
        target_idx = label_hat.detach()

        # 1) MProtoNet attribution 
        attr = attribute_mprotonet_2d(
            self.net,
            image,                 # [B,4,H,W]
            target_idx,
            normalize=True
        )                           # -> [B,4,H,W]

        
        # # If Grad Cam attr to be used: IMPORTANT: Lightning test/val runs in no_grad, but GradCAM needs grad.
        # with torch.enable_grad():
        #     attr = attribute_gradcam_2d(
        #         self.net,
        #         image,
        #         target_idx,
        #         normalize=True
        #     )

        # ##sanity check
        # print("attr min/max:", float(attr.min()), float(attr.max()), "mean:", float(attr.mean()))
        # print("nonzero ratio:", float((attr > 0.05).float().mean()))


        # attr: [B,4,H,W]
        attr = (attr - attr.amin(dim=(2,3), keepdim=True)) / (
            attr.amax(dim=(2,3), keepdim=True) - attr.amin(dim=(2,3), keepdim=True) + 1e-8
        )
        if attr.shape[1] > 1:   ## If 4 channels from mproto and not 1 from grad cam
            attr = attr.mean(dim=1, keepdim=True)


        # store continuous attribution + GT for unified evaluation
        # attr: [B,1,H,W] -> [B,H,W]
        attr = torch.rot90(attr, k=1, dims=(-2, -1))    # rotate pred_mask 90° left    ####TODO
        self._test_attr_maps.append(attr.detach().cpu()[:, 0])  # [B,H,W]

        # binary mask is [B,1,H,W] (or sometimes [B,H,W])
        if mask.dim() == 4:
            wt = (mask[:, 0] > 0).detach().cpu()               # [B,H,W]
        else:
            wt = (mask > 0).detach().cpu()                     # [B,H,W]
        self._test_gt_wt.append(wt)

        # store labels (for optional per-class stats)
        if label.ndim > 1:
            y_idx = label.argmax(1).detach().cpu()
        else:
            y_idx = label.detach().cpu()
        self._test_y.append(y_idx)


        # 3) Prepare segmentation mask
        seg = mask                 # [B,1,H,W]

        # 4) Location coherence (repo metric)
        lc_ap  = lc_2d(attr, seg, threshold=0.5, annos=None, metric="AP")
        lc_dsc = lc_2d(attr, seg, threshold=0.5, annos=None, metric="DSC")

        self._lc_ap_vals.append(lc_ap.detach().cpu())
        self._lc_dsc_vals.append(lc_dsc.detach().cpu())


        return loss


    def predict_step(self, batch, _):
        image, _, _ = batch
        if self.hparams.model_segclassify:
            pred, pred_mask = self(image)
            pred_mask = self.mask_merge(pred_mask, pred)
            # THRESHOLD FOR BOTH MODELS
            pred_mask = (pred_mask > 0.5).float()
        else:
            pred, pred_mask = self(image), None
        return torch.argmax(pred, dim=1), pred_mask

    def prepare_dataset_splits(self):


        csv_path = Path(self.hparams.main_path) / "BraTS_2020/MICCAI_BraTS2020_TrainingData/name_mapping.csv"
        df = pd.read_csv(csv_path, usecols=["Grade"])
        # df index is subjectNumber-1 in your stacking script
        self._grade_by_subject_number = {i+1: df.Grade[i] for i in range(len(df))}


        #testing for review
        PICKLE_DIR = Path(f"{self.hparams.main_path}/Dataset/PreProcessed_Brats2020/pickles/full/Axi")

        with open(PICKLE_DIR / "X_allNonEmptySlices.pickle", "rb") as f:
            X = pickle.load(f)

        with open(PICKLE_DIR / "m_allNonEmptySlices.pickle", "rb") as f:
            m = pickle.load(f)

        with open(PICKLE_DIR / "y_allNonEmptySlices.pickle", "rb") as f:
            y = pickle.load(f)

        ids = pickle.load(open(PICKLE_DIR/"ids_allNonEmptySlices.pickle","rb"))

        print("Loaded pickles:")
        print("X shape:", X.shape)  # (N, 4, H, W)
        print("m shape:", m.shape)  # (N, 1, H, W)
        print("y shape:", y.shape)  # (N,)
        print("Classes present:", np.unique(y))

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


        self.ind_trainval = ind_trainval
        self.ind_test = ind_test





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
        base = TumourDataset(
            self.hparams.contrast, self.X_test, self.m_test, self.y_test,
            maxinorm=True if self.hparams.normmode == 3 else False,
            transforms=transforms.Compose(self.trans),
            trans=transforms.Compose(self.trans),
        )
        ##to link slices to global indices for saving predictions to aggregate analysis to per patient level
        ds = WithGlobalIndex(base, global_indices=self.ind_test)  # <<<<<<<<<<
        return DataLoader(ds, shuffle=False, batch_size=self.hparams.batch_size,
                        pin_memory=True, num_workers=self.hparams.workers)



    def on_train_epoch_end(self):
        if len(self._train_losses) > 0:
            avg_loss = torch.stack(self._train_losses).median()
            self.log('training_loss', avg_loss, prog_bar=True)
        self._train_losses.clear()


    def on_validation_epoch_end(self):
        if len(self._val_losses) > 0:
            avg_loss = torch.stack(self._val_losses).median()
            self.log('val_loss', avg_loss, prog_bar=True)

        if self._val_count > 0:
            val_acc = self._val_correct / float(self._val_count)
            self.log('accuracy', val_acc, prog_bar=True)

        if self.hparams.model_segclassify and len(self._val_dices) > 0:
            dice = torch.stack(self._val_dices).median()
            self.log('dice', dice, prog_bar=True)

        # log only every 10 epochs (matches push cadence)
        if (self.current_epoch + 1) % 10 == 0:
            self._log_proto_selectivity(self.val_dataloader(), max_batches=10, prefix="val")


        # reset buffers
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

        image_np = images.squeeze().numpy() if len(images) > 0 else None
        mask = torch.cat(self._test_masks).squeeze().numpy()

        out_dir = os.path.join(self.hparams.out_path, "Results", self.hparams.trainID)

        # if len(self._lc_ap_vals) > 0:
        #     lc_ap_mean = torch.cat(self._lc_ap_vals).mean().item()
        #     self.log("LC_AP_WT_Th0.5", lc_ap_mean)

        # if len(self._lc_dsc_vals) > 0:
        #     lc_dsc_mean = torch.cat(self._lc_dsc_vals).mean().item()
        #     self.log("LC_DSC_WT_Th0.5", lc_dsc_mean)

        # print(f"[Test] LC(AP): {lc_ap_mean:.4f} | LC(DSC): {lc_dsc_mean:.4f}")

        # with open(os.path.join(out_dir, "interpretability.txt"), "a") as f:
        #     f.write(f"LC_AP_WT_Th0.5: {lc_ap_mean:.6f}\n")
        #     f.write(f"LC_DSC_WT_Th0.5: {lc_dsc_mean:.6f}\n")

        analysis = False
        # Explainability (Top-k or multi-otsu) -> ONLY 2 CSVs (run once only)
        if analysis:
            if len(self._test_attr_maps) > 0:

                attr_all = torch.cat(self._test_attr_maps, dim=0).numpy()  # (N,H,W)
                gt_all   = torch.cat(self._test_gt_wt, dim=0).numpy()      # (N,H,W)
                y_all    = torch.cat(self._test_y, dim=0).numpy()          # (N,)
                gidx_all = torch.cat(self._test_gidx, dim=0).cpu().numpy().astype(int)  # (N,)

                out_dir = os.path.join(self.hparams.out_path, "Results", self.hparams.trainID)
                fold_id = int(self.hparams.foldID) if hasattr(self.hparams, "foldID") else 0
                os.makedirs(out_dir, exist_ok=True)

                # Use global indices as slice ids (best for tracing)
                slice_ids = gidx_all

                per_slice_rows = []
                summary_rows = []

                # filter: ONLY slices with true label = 1 (LGG) or 2 (HGG)
                keep_idx = np.where(np.isin(y_all.astype(int), [1, 2]))[0]
                print(f"[Expl] tumor-only slices (label 1/2): {len(keep_idx)} / {attr_all.shape[0]}")

                # method = "topk"
                method = "multiotsu"
                # method = "fixed_thr"
                if method == "topk":
                    ks = [0.01,0.02, 0.05, 0.10]
                elif method == "multiotsu":
                    ks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1, 1.6]
                elif method == "fixed_thr":
                    ks = [0.5]
                norm = True

                for k in ks:
                    dice_list, ap_list, rec_list, fp_list = [], [], [], []

                    for i in keep_idx:
                        m = eval_explanation_map(
                            x_map=attr_all[i],
                            gt_mask=gt_all[i],
                            method=method,
                            k=k,
                            normalize=norm,
                            fixed_thr=k,
                        )

                        d = float(m["dice_L"])
                        a = float(m["AP"])
                        r = float(m["recall_L"])
                        f = float(m["FP_L"])

                        dice_list.append(d)
                        ap_list.append(a)
                        rec_list.append(r)
                        fp_list.append(f)

                        per_slice_rows.append({
                            "fold": fold_id,
                            "slice_idx": int(slice_ids[i]),
                            "true_label": int(y_all[i]),
                            "k": float(k),
                            "dice_L": d,
                            "AP": a,
                            "recall_L": r,
                            "FP_L": f,
                            "gt_pixels": int(gt_all[i].sum()),
                        })

                    # median + IQR
                    dice_med, dice_iqr, dice_q1, dice_q3 = median_iqr(dice_list)
                    ap_med, ap_iqr, ap_q1, ap_q3         = median_iqr(ap_list)
                    rec_med, rec_iqr, rec_q1, rec_q3     = median_iqr(rec_list)
                    fp_med, fp_iqr, fp_q1, fp_q3         = median_iqr(fp_list)

                    summary_rows.append({
                        "k": float(k),

                        "dice_L_median": dice_med,
                        "dice_L_IQR": dice_iqr,
                        "dice_L_Q1": dice_q1,
                        "dice_L_Q3": dice_q3,

                        "AP_median": ap_med,
                        "AP_IQR": ap_iqr,
                        "AP_Q1": ap_q1,
                        "AP_Q3": ap_q3,

                        "recall_L_median": rec_med,
                        "recall_L_IQR": rec_iqr,
                        "recall_L_Q1": rec_q1,
                        "recall_L_Q3": rec_q3,

                        "FP_L_median": fp_med,
                        "FP_L_IQR": fp_iqr,
                        "FP_L_Q1": fp_q1,
                        "FP_L_Q3": fp_q3,

                        "n_slices": int(len(dice_list)),
                    })

                # Save the ONLY 2 files you want
                pd.DataFrame(summary_rows).to_csv(
                    os.path.join(out_dir, f"explainability_{method}_medianIQR_fold{fold_id}_norm:{norm}.csv"),
                    index=False
                )
                pd.DataFrame(per_slice_rows).to_csv(
                    os.path.join(out_dir, f"explainability_{method}_perSlice_fold{fold_id}_norm:{norm}.csv"),
                    index=False
                )

                # optional console print
                for row in summary_rows:
                    print(
                        f"[Expl] k={row['k']:.2f} | "
                        f"DSC={row['dice_L_median']:.4f} ± {row['dice_L_IQR']:.4f} | "
                        f"AP={row['AP_median']:.4f} ± {row['AP_IQR']:.4f} | "
                        f"Rec={row['recall_L_median']:.4f} ± {row['recall_L_IQR']:.4f} | "
                        f"FP={row['FP_L_median']:.4f} ± {row['FP_L_IQR']:.4f}"
                    )

        ##hard coded best offset for mproto from the previous analysis
        offset = 0.5

        result_analyser(
            preds,
            y_all,
            attr_all,
            gt_all,
            self.classes,
            out_dir,
            image_np,
            offset,
            self.hparams.out_act,
            self.hparams.network,
        )

        # Save fold predictions
        fold_id = int(self.hparams.foldID) if hasattr(self.hparams, "foldID") else 0
        save_fold_predictions(labels.numpy(), probs.numpy(), out_dir, fold_id)

        # reset buffers
        self._test_losses.clear()
        self._test_preds.clear()
        self._test_labels.clear()
        self._test_probs.clear()
        self._test_images.clear()
        self._test_pred_masks.clear()
        self._test_masks.clear()
        self._lc_ap_vals.clear()
        self._lc_dsc_vals.clear()
        self._test_attr_maps.clear()
        self._test_gt_wt.clear()
        self._test_y.clear()




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


class MProtoPushCallback(pl.Callback):
    def __init__(self, push_fn, every=10, start=10, last_steps=10):
        self.push_fn = push_fn
        self.every = every
        self.start = start
        self.last_steps = last_steps

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if pl_module.hparams.network != "MProtoNet2D":
            return
        if epoch < self.start or epoch % self.every != 0:
            return

        # 1) PUSH
        pl_module.net.eval()
        device = pl_module.device
        self.push_fn(dataloader=pl_module.train_dataloader(), model=pl_module.net, device=device)

        # log prototype deltas (stability)
        if hasattr(pl_module.net, "prototype_last_deltas"):
            deltas = pl_module.net.prototype_last_deltas
            pl_module.log("push_delta_mean", deltas.mean(), prog_bar=True)
            pl_module.log("push_delta_max", deltas.max(), prog_bar=False)

        # log prototype identity (source stability)
        if hasattr(pl_module.net, "prototype_sources"):
            # simple signature per prototype (batch_idx + sample_in_batch)
            sig = []
            for k, s in enumerate(pl_module.net.prototype_sources):
                if s is None:
                    sig.append(("none", "none"))
                else:
                    sig.append((s["batch_idx"], s["sample_in_batch"]))
            # store on module to compare next push
            if not hasattr(pl_module, "_prev_proto_sig"):
                pl_module._prev_proto_sig = sig
                stable = 0.0
            else:
                stable = sum([a == b for a, b in zip(pl_module._prev_proto_sig, sig)]) / float(len(sig))
                pl_module._prev_proto_sig = sig

            pl_module.log("push_identity_stability", stable, prog_bar=True)

        # 2) LAST-LAYER TRAINING (10 loops) like repo
        pl_module.train_last_layer_loops(n_loops=self.last_steps)






def main():
    torch.set_num_threads(1)
    resume_from_checkpoint = True
    chkp_type = "Best"  #"Best"  #"Last"
    wnbactive = False
    out_act: str = "sigmoid"  #softmax or None or "sigmoid" or "relu"
    network: str = "MProtoNet2D"  #"GP_UNet" #"resnet18" # "GP_reconresnt"
    num_epochs: int = 300     #150
    run_prefix=f"Org_{network}_{num_epochs}e"    # _wnbact_deter  _2  _bilinear _sinc
    use_amp: bool = True     #Automatic Mixed Precission
    lr:float = 1e-3    #found by findlr on pc 4.786300923226383e-07    #Default: 1e-3
    batch_size:int = 32      #for Seg batch size 8 and less succeded acording to autoScaleBatchSize
    accumulate_grad_batches: int = 16        #248*249 = 32 or 64 for all models and orientations
    workers: int = 2        #stimulate 4 #GPU18 8
    normmode: int = 3 #0: ZNorm, 1: Divide by nth percentile, 2: Divide by Max
    percentile: int = 99 #nth percentile to be used only for normmode 1
    depth: int = 3    #for GP_UNet depth: default = 3, can go deeper or shallower #5 is the depth of original GPUNet
    wf: int = 6       #for GP_UNet no of filters in each layer = 2**wf  default: 6  meaning starting 64 then 128, 256 till depth 3
    model_segclassify: bool = False         # when model is GP- then true
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

    trainID = run_prefix + "_" + Dataset + "_" + amount + "_" + contrast + "_" + orient + "_" + network + "_"  + str(g_c_val) + "-" + str(g_c_algo) + "_w_d-" + str(w_d) + "_drop-" + str(dropout) + "_" + upalgo + \
            "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds)


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

    model = Classifier(**vars(hparams))

    checkpoint_callback = ModelCheckpoint(
        dirpath=pjoin(hparams.out_path, "Checkpoints", hparams.trainID),
        monitor='val_loss',
        save_last=True,
    )
    logger = WandbLogger(name=trainID, id=trainID, project='MasterThesis_Hadya',
                            group='Baseline', entity='mickchimp', config=hparams)
    logger.watch(model, log='all', log_freq=100)

    #specify which checkpoint
    if resume_from_checkpoint:
        if chkp_type == "Best":
            if network == "MProtoNet2D":
                chkpoint = "/project/Data/GPModels/outputs/Checkpoints/Org_MProtoNet2D_300e_Brats20_full_allCont_Axi_MProtoNet2D_1-value_w_d-0.0005_drop-True_sinc_norm3_fold0of5/epoch=274-step=202125.ckpt"
            # checkpoint_dir = pjoin(hparams.out_path, "Checkpoints", hparams.trainID)
            # chkpoint = pjoin(checkpoint_dir, sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1])
        elif chkp_type == "Last":
            chkpoint = pjoin(hparams.out_path, "Checkpoints", hparams.trainID, "last.ckpt")

    else:
        chkpoint = None

    if not changeAnyTesthparam:
        # os.makedirs(pjoin(hparams.out_path, "Results", hparams.trainID), exist_ok=True)
        os.makedirs(pjoin("/home/hadya/GPModels/output", "Results", hparams.trainID), exist_ok=True)

    use_gpu = torch.cuda.is_available()

    # train
    trainer = Trainer(
        logger=logger,
        precision="16-mixed" if (use_gpu and use_amp) else 32,  # avoid precision=16 warning
        # gpus=1,     #gpus=1   ##deprecated
        accelerator="gpu" if use_gpu else "cpu",  ##instead of gpus=1
        devices=1 if use_gpu else "auto",          ##instead of gpus=1
        inference_mode=False,
        # checkpoint_callback=True,  #checkpoint_callback=checkpoint_callback,   was for old trainer.py istead of 326, 327 # remove this, use callbacks list instead:
        callbacks=[
            checkpoint_callback,
            MProtoPushCallback(push_fn=push_prototypes_2d, every=10, start=10, last_steps=10),
        ],
        max_epochs=hparams.max_epochs,
        # terminate_on_nan=True,       
        deterministic=True,
        # accumulate_grad_batches=accumulate_grad_batches,
        # resume_from_checkpoint=chkpoint,        # deprecated: handle when calling trainer.fit(...)
        # auto_scale_batch_size='binsearch' if autoScaleBatchSize else None,
        # auto_lr_find=findLR,
        # gradient_clip_val=g_c_val,
        # gradient_clip_algorithm=g_c_algo
    )

    # Only train
    if runMode == "onlyTrain":
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
            trainer.test(model, test_dataloaders=model.test_dataloader())

        #2- resume or Train and Test from scratch without changing Testhparam
        else:  # not changeAnyTesthparam
            if chkpoint is not None:
                trainer.fit(model, ckpt_path=chkpoint)
            else:
                trainer.fit(model)

            trainer.test(test_dataloaders=model.test_dataloader())

if __name__ == '__main__':
    main()
