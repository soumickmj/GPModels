import os
import pickle
import sys
from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torchio as tio
import torchvision.models as models
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from model.tvmodels import TVModelWrapper 
from model.Res101 import ResNet101 
from model.Res152 import ResNet152 
from model.Res50 import ResNet50 
from model.Res34 import ResNet34 
from model.ReconResNet.GP_ReconResNet import GP_ReconResNet
from model.Unet.GP_UNet import GP_UNet
from utilities.dataset import TumourDataset
from utilities.utils import (Dice, fromTorchIO, getValStat, result_analyser,
                             toTorchIO)

seed_everything(1701)

class Classifier(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.prepare_dataset_splits()
        if len(self.classIDs) == 3:
            self.classes = ['Meningioma', 'Glioma', 'Pitutary']
        elif len(self.classIDs) == 4:
            self.classes = ['Healthy', 'Meningioma', 'Glioma', 'Pitutary']
        else:
            sys.exit("Number of classes is other than 3 or 4")   

        tio_trans = {                    
                    tio.RandomMotion(): 1, 
                    tio.RandomGhosting(): 1,
                    tio.RandomBiasField(): 1,
                    tio.RandomBlur(): 1,
                    tio.RandomNoise(): 1,
                }
        self.tio_trans = transforms.Compose([
                                                toTorchIO(),
                                                tio.Compose([tio.OneOf(tio_trans, p=0.5)]),
                                                fromTorchIO()
                                            ])      

        self.trans_aug = [
                            transforms.ToPILImage(),                                          
                            transforms.RandomRotation((-330, +330)),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                        ]       
        self.trans = [ transforms.ToTensor(), ]
        if kwargs['normmode'] == 0: #Z-Score normalisation
            self.trans.append(transforms.Normalize([self.DSmean], [self.DSstd])) 
        elif kwargs['normmode'] == 1: #Divide by nth percentile (default: 99th)
            self.trans.append(transforms.Normalize([0], [self.DSperval]))
        elif kwargs['normmode'] == 2: #Divide by Max
            self.trans.append(transforms.Normalize([0], [self.DSmax]))
            #self.trans.append(transforms.Normalize([self.DSmin], [(self.DSmax-self.DSmin/255-0)]))

        if kwargs['network'] == "resnet18":
            self.net = TVModelWrapper(model=models.resnet18, num_classes=len(self.classIDs))
        elif kwargs['network'] == "resnet101":
            self.net = ResNet101(model=models.resnet101, num_classes=len(self.classIDs))
        elif kwargs['network'] == "resnet152":
            self.net = ResNet152(model=models.resnet152, num_classes=len(self.classIDs))
        elif kwargs['network'] == "resnet50":
            self.net = ResNet50(model=models.resnet50, num_classes=len(self.classIDs))
        elif kwargs['network'] == "resnet34":
            self.net = ResNet34(model=models.resnet34, num_classes=len(self.classIDs))
        elif kwargs['network'] == "GP_UNet":
            self.net = GP_UNet(n_classes=len(self.classIDs))
        elif kwargs['network'] == "GP_ReconResNet":
            self.net = GP_ReconResNet(n_classes=len(self.classIDs))
        self.loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.classWs).float() if kwargs['useClassWeight'] else None)
        self.mask_accuracy = Dice()    

    def mask_merge(self, mask, pred=None):
        if self.hparams.MaskMergeMode == "sum":
            return torch.sum(mask, dim=1, keepdims=True)
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
        )
        return {
            'optimizer': optimizer,
            'monitor': 'val_loss',
        }

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        image, _, label = batch
        out = self(image)
        loss = self.loss(out, label)
        if loss != loss:  #when loss = nan, loss is not a number so it is not equal itself
            print("loss is nan")
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, _):
        image, mask, label = batch
        if self.hparams.model_segclassify:
            out, pred_mask = self(image)
            pred_mask = self.mask_merge(pred_mask, out)
            _, dice = self.mask_accuracy(pred_mask, mask)
        else:
            out, pred_mask, dice = self(image), None, -1
        loss = self.loss(out, label)
        label_hat = torch.argmax(out, dim=1)
        n_correct = torch.sum(label == label_hat).item()        
        return {'val_loss': loss, 'n_correct':n_correct, 'dice':dice}

    def test_step(self, batch, _):
        image, mask, label = batch
        if self.hparams.model_segclassify:
            pred, pred_mask = self(image)
            pred_mask = self.mask_merge(pred_mask, pred).cpu()
        else:
            pred, pred_mask = self(image), None       
        loss = self.loss(pred, label)
        label_hat = torch.argmax(pred, dim=1)
        return {'test_loss': loss.cpu(), 'pred':label_hat.cpu(), 
                'label':label.cpu(), 'pred_mask':pred_mask, 'mask':mask.cpu()}

    def predict_step(self, batch, _):
        image, _, _ = batch
        if self.hparams.model_segclassify:
            pred, pred_mask = self(image)
            pred_mask = self.mask_merge(pred_mask, pred)
        else:
            pred, pred_mask = self(image), None 
        return torch.argmax(pred, dim=1), pred_mask

    def prepare_dataset_splits(self):
        data_seg = pickle.load(
            open(
                f'{self.hparams.main_path}/dataset/training_data_{self.hparams.orient}.pickle',
                'rb',
            )
        )

        X = data_seg[:,0,...]
        m = data_seg[:,1,...]
        y = pickle.load(
            open(
                f'{self.hparams.main_path}/dataset/labels_{self.hparams.orient}.pickle',
                'rb',
            )
        )


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
        self.DSmean, self.DSstd, self.DSperval, self.DSmax, self.DSmin = X.mean(), X.std(), np.percentile(X, self.hparams.percentile), X.max(), X.min()

        ind_train, ind_val  = list(StratifiedShuffleSplit(n_splits=self.hparams.n_folds, test_size=self.hparams.val_percent, random_state=42).split(X, y))[self.hparams.foldID]
        self.X_train = X[ind_train]
        self.y_train = y[ind_train]
        self.m_train = m[ind_train]
        self.X_val = X[ind_val]
        self.y_val = y[ind_val]
        self.m_val = m[ind_val]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(TumourDataset(self.X_train, self.m_train, self.y_train, 
                          transforms=transforms.Compose([self.tio_trans]+self.trans_aug+self.trans if self.hparams.MRIAug else self.trans_aug+self.trans)),
                          shuffle=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(TumourDataset(self.X_val, self.m_val, self.y_val, transforms=transforms.Compose(self.trans)),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(TumourDataset(self.X_test, self.m_test, self.y_test, transforms=transforms.Compose(self.trans)),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).median()
        self.log('training_loss', avg_loss)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).median()
        self.log('val_loss', avg_loss)
        n_correct = sum(x['n_correct'] for x in outputs)
        val_acc = n_correct / (len(self.X_val) * 1.0)
        self.log('accuracy', val_acc)
        if self.hparams.model_segclassify:
            dice = torch.stack([x['dice'] for x in outputs]).median()
            self.log('dice', dice)        

    def test_epoch_end(self, outputs: List[Any]) -> None:
        test_loss = torch.stack([x['test_loss'] for x in outputs]).median()
        self.log('test_loss', test_loss)

        pred = torch.cat([x['pred'] for x in outputs])
        labels = torch.cat([x['label'] for x in outputs])
        test_acc = torch.sum(labels == pred).item() / (len(labels) * 1.0)
        self.log('test_acc', test_acc)

        if self.hparams.model_segclassify:
            mask = torch.cat([x['mask'] for x in outputs]).squeeze().numpy()
            pred_mask = torch.cat([x['pred_mask'] for x in outputs]).squeeze().numpy()
        else:
            mask, pred_mask = None, None

        result_analyser(pred, labels, pred_mask, mask, self.classes, os.path.join(self.hparams.out_path, "Results", self.hparams.trainID))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--main_path', type=str)
        parser.add_argument('--out_path', type=str)
        parser.add_argument('--orient', type=str)
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
        return parser

def main():
    torch.set_num_threads(1)
    resume_from_checkpoint = False
    chkp_type = "Best"  #"Best"  #"Last"
    wnbactive = False
    run_prefix="Initial"
    num_epochs: int = 1
    use_amp: bool = True
    lr:float = 1e-3
    batch_size:int = 4
    accumulate_grad_batches: int = 2
    workers: int = 0
    normmode: int = 0 #0: ZNorm, 1: Divide by nth percentile, 2: Divide by Max
    percentile: int = 99 #nth percentile to be used only for normmode 1
    network: str = "GP_UNet" #"resnet18" 
    model_segclassify: bool = True
    useClassWeight: bool = True 
    main_path: str = r"D:\Rough\H\brainTumorDataPublic_China"
    out_path: str = r"D:\Rough\H"
    orient: str = "All"
    test_percent: float = 0.25
    val_percent: float = 0.20
    n_folds: int = 5
    foldID: int = 0
    MRIAug: bool = True
    MaskMergeMode: str = "sum" #sum or predselect
    autoScaleBatchSize = False
    findLR = False
    runMode = "TrainTest"    # "onlyTrain" # "onlyTest" # "TrainTest"
    changeAnyTesthparam = False  #True when wanting to change any Test hparams in a saved checkpiont when resuming TrainTest or onlyTest


    if MaskMergeMode == "sum":
        ChngdMaskMergeMode = "predselect"  
    elif MaskMergeMode == "predselect":
        ChngdMaskMergeMode = "sum"  

    trainID = run_prefix + "_" + orient + "_" + network + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
                "_" + ("WMRIAug" if MRIAug else "WoMRIAug") + ("_Mask"+MaskMergeMode if model_segclassify else "") +\
                "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds) 

    parser = ArgumentParser()
    parser = Classifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.lr = lr
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
    hparams.orient = orient
    hparams.test_percent = test_percent
    hparams.val_percent = val_percent
    hparams.n_folds = n_folds
    hparams.foldID = foldID
    hparams.MRIAug = MRIAug
    hparams.MaskMergeMode = MaskMergeMode
    hparams.autoScaleBatchSize = autoScaleBatchSize
    hparams.findLR = findLR



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
            checkpoint_dir = pjoin(hparams.out_path, "Checkpoints", hparams.trainID)
            chkpoint = pjoin(checkpoint_dir, sorted([x for x in os.listdir(checkpoint_dir) if "epoch" in x])[-1])
        elif chkp_type == "Last":
            chkpoint = pjoin(hparams.out_path, "Checkpoints", hparams.trainID, "last.ckpt")
    else:
        chkpoint = None

    if not changeAnyTesthparam:
        os.makedirs(pjoin(hparams.out_path, "Results", hparams.trainID), exist_ok=True)
    
    # train
    trainer = Trainer(
        logger=logger,
        precision=16 if use_amp else 32,
        gpus=1,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        max_epochs=hparams.max_epochs,
        terminate_on_nan=True,
        deterministic=True,
        accumulate_grad_batches=accumulate_grad_batches,
        resume_from_checkpoint=chkpoint,
        auto_scale_batch_size='binsearch' if autoScaleBatchSize else None,
        auto_lr_find=findLR
    )

    
    # Only train
    if runMode == "onlyTrain":
        if autoScaleBatchSize or findLR:
            trainer.tune(model)

        trainer.fit(model)


    # Only test using the best model!
    elif runMode == "onlyTest":

        #load checkpoint
        if chkpoint == None:
            print("Warning!! the next line will produce an error, resume_from_Checkpoint is set to False")
        model = Classifier.load_from_checkpoint(chkpoint)
        #If there us any change in Testhparam, specify it below
        if changeAnyTesthparam:
            model.hparams.MaskMergeMode = ChngdMaskMergeMode
            model.hparams.trainID = run_prefix + "_" + orient + "_" + network + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
                    "_" + ("WMRIAug" if MRIAug else "WoMRIAug") + ("_Mask_chng_"+model.hparams.MaskMergeMode if model_segclassify else "") +\
                    "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds) 
        
        #onlyTest
        trainer.test(model, test_dataloaders=model.test_dataloader())


    #else training and testing (resuming training or from scratch)
    elif runMode == "TrainTest":
        if autoScaleBatchSize or findLR:
            trainer.tune(model)

        #1-Resume training with changing Testing hparam
        if changeAnyTesthparam:

            #Train
            trainer.fit(model)

            #load checkpoint
            if chkpoint == None:
                print("Warning!! the next line will produce an error, resume_from_Checkpoint is set to False")
            model = Classifier.load_from_checkpoint(chkpoint)
            #specify the change in Test hParameters below
            model.hparams.MaskMergeMode = ChngdMaskMergeMode
            #specify the change "chng" in train id for chngd output folder name
            model.hparams.trainID = run_prefix + "_" + orient + "_" + network + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
                    "_" + ("WMRIAug" if MRIAug else "WoMRIAug") + ("_Mask_chng_"+model.hparams.MaskMergeMode if model_segclassify else "") +\
                    "_norm" + str(normmode) + ("" if normmode!=1 else "p"+str(percentile)) + "_fold" + str(foldID) + "of" + str(n_folds) 
            
            #Test
            trainer.test(model, test_dataloaders=model.test_dataloader())

        #2- resume or Train and Test from scratch without changing Testhparam
        elif not changeAnyTesthparam:
            trainer.fit(model)
            trainer.test(test_dataloaders=model.test_dataloader())
        

if __name__ == '__main__':
    main()
