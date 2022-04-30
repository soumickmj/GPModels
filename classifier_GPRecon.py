import os
import pickle
import sys
from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
#from torch.utils.data.dataset import Dataset
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
from model.ReconResNet.GP_ReconResNet import GP_ReconResNet
from model.Unet.GP_UNet import GP_UNet
from utilities.dataset import TumourDataset
from utilities.utils import (Dice, fromTorchIO, getValStat, result_analyser,
                             toTorchIO)
from utilities.load import load_Brats_full





seed_everything(1701, workers=True)

class Classifier(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.prepare_dataset_splits()
        if self.hparams.Dataset == "China" or "Initial" in self.hparams.trainID:
            if len(self.classIDs) == 3:
                self.classes = ['Meningioma', 'Glioma', 'Pitutary']
            elif len(self.classIDs) == 4:
                self.classes = ['Healthy', 'Meningioma', 'Glioma', 'Pitutary']
            else:
                sys.exit("Number of classes is other than 3 or 4")   

        elif self.hparams.Dataset == "Brats20":
            if len(self.classIDs) == 3:
                self.classes = ['Healthy', 'LGG', 'HGG']
            else:
                sys.exit("Number of classes is other than 3")                       

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

        self.transform2 = [self.tio_trans] + self.trans_aug + self.trans

        if kwargs['normmode'] == 0: #Z-Score normalisation
            self.trans.append(transforms.Normalize([self.DSmean], [self.DSstd])) 
        elif kwargs['normmode'] == 1: #Divide by nth percentile (default: 99th)
            self.trans.append(transforms.Normalize([0], [self.DSperval]))
        elif kwargs['normmode'] == 2: #Divide by Max
            self.trans.append(transforms.Normalize([0], [self.DSmax]))
            #self.trans.append(transforms.Normalize([self.DSmin], [(self.DSmax-self.DSmin/255-0)]))

        if kwargs['network'] == "resnet18":
            self.net = TVModelWrapper(model=models.resnet18, num_classes=len(self.classIDs))
        elif kwargs['network'] == "GP_UNet":
            self.net = GP_UNet(n_classes=len(self.classIDs), up_mode=self.hparams.upalgo)
        elif kwargs['network'] == "GP_ReconResNet":
            # self.net = GP_ReconResNet(n_classes=len(self.classIDs)) only used with old chkp resume doesn't have up algo in name
            self.net = GP_ReconResNet(in_channels= (4 if self.hparams.contrast == "allCont" else 1), n_classes=len(self.classIDs), res_drop_prob=self.hparams.dropout, out_act=self.hparams.out_act, upinterp_algo=self.hparams.upalgo)
            
        self.loss = nn.CrossEntropyLoss(weight=torch.from_numpy(self.classWs).float() if kwargs['useClassWeight'] else None)
        self.mask_accuracy = Dice()        








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
                'label':label.cpu(), 'pred_mask':pred_mask, 'mask':mask.cpu(), 'image':image.cpu()}

    def predict_step(self, batch, _):
        image, _, _ = batch
        if self.hparams.model_segclassify:
            pred, pred_mask = self(image)
            pred_mask = self.mask_merge(pred_mask, pred)
        else:
            pred, pred_mask = self(image), None 
        return torch.argmax(pred, dim=1), pred_mask

    def prepare_dataset_splits(self):

        if self.hparams.Dataset == "Brats20" and self.hparams.amount == "full":
            X, m, y = load_Brats_full(self.hparams.amount, self.hparams.orient, (self.hparams.main_path + "/Brats2020/Dataset"), self.hparams.contrast)

        else: 

            if  "Initial" in self.hparams.trainID or self.hparams.Dataset == "China":
                data_seg = pickle.load(open(self.hparams.main_path+'/dataset/training_data_{}.pickle'.format(self.hparams.orient), 'rb'))
                y = pickle.load(open(self.hparams.main_path+'/dataset/labels_{}.pickle'.format(self.hparams.orient), 'rb'))
            elif self.hparams.Dataset == "Brats20":
                data_seg = pickle.load(open(self.hparams.main_path+f'/Brats2020/Dataset/pickles/{self.hparams.amount}/{self.hparams.orient}/{self.hparams.contrast}_training_data.pickle', 'rb'))
                y = pickle.load(open(self.hparams.main_path+f'/Brats2020/Dataset/pickles/{self.hparams.amount}/{self.hparams.orient}/{self.hparams.contrast}_labels.pickle', 'rb'))

            X = data_seg[:,0,...]
            m = data_seg[:,1,...]
        
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
        return DataLoader(TumourDataset(self.hparams.contrast, self.X_test, self.m_test, self.y_test, maxinorm=True if self.hparams.normmode == 3 else False, transforms=transforms.Compose(self.trans), trans=transforms.Compose(self.trans)),
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True, num_workers=self.hparams.workers)    #, collate_fn=collate_fn

    def training_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).median()
        self.log('training_loss', avg_loss)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).median()
        self.log('val_loss', avg_loss)
        n_correct = sum([x['n_correct'] for x in outputs])
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

        image = torch.cat([x['image'] for x in outputs]).squeeze().numpy()

        result_analyser(pred, labels, pred_mask, mask, self.classes, os.path.join(self.hparams.out_path, "Results", self.hparams.trainID), image)

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
        parser.add_argument('--dropout', type=float)
        return parser





def main():
    torch.set_num_threads(1)
    resume_from_checkpoint = True
    chkp_type = "Last"  #"Best"  #"Last"
    wnbactive = True
    out_act: str = "sigmoid"    #softmax or None or "sigmoid" or "relu"
    # run_prefix="InitialGPRecon_150epoch_wnbactive_deterministic"    #_bilinear _sinc
    # run_prefix="GPRecon_150e_wnbact_deter"    #   _2  _bilinear _sinc
    run_prefix=f"GPRecon_150e_{out_act}"    #_wnbact_deter _bilinear _sinc
    num_epochs: int = 150     #150
    use_amp: bool = True       #Automatic Mixed Precission
    lr:float = 1e-3    #found by findlr on pc 4.786300923226383e-07    #Default: 1e-3
    batch_size:int = 4      #for Seg batch size 8 and less succeded acording to autoScaleBatchSize
    accumulate_grad_batches: int = 32        #248*249 = 32 or 64 for all models and orientations
    workers: int = 2        #stimulate 4 #GPU18 8
    normmode: int = 3 #0: ZNorm, 1: Divide by nth percentile, 2: Divide by dataset Max # 3: devide by individual Max #None = 5
    percentile: int = 99 #nth percentile to be used only for normmode 1
    network: str = "GP_ReconResNet"  #"GP_UNet" #"resnet18" # "GP_reconresnt"
    model_segclassify: bool = True         # when model is GP- then true
    useClassWeight: bool = True 
    main_path: str = '/mnt/public/hadya/master_Yassin'
    out_path: str = '/mnt/public/hadya/master_Yassin/output'
    Dataset: str = "Brats20"     #China or Brats20
    dropout: float = 0.5 if Dataset == "Brats20" else 0.2     # default=0.2, or more 0.5
    w_d:float = 5e-4 if Dataset == "Brats20" else 0     # 5e-4 proposed for CNN and 0 for no regulirization
    g_c_val: int = 1 if Dataset == "Brats20" else 0
    g_c_algo: str = 'value'   #'norm' by default, or set to "value"
    contrast: str = "allCont"     #t1, t1ce, t2, flair, allCont 
    orient: str = "Axi"     #All Sag Cor Axi
    amount: str = "full"     #amount taken from complete dataset full, half or 100 
    test_percent: float = 0.25
    val_percent: float = 0.20
    n_folds: int = 5
    foldID: int = 0
    MRIAug: bool = False
    upalgo: str = "sinc"      #GPRecon "upconv" = transconv or "bilinear" or "sinc" = two diff interpolate algo for upsampling (conve + interpolate)  #GPUNet "upconv" =convtrans, "upsample" = conv + interpolate(bilinear)   
    MaskMergeMode: str = "predselect" #sum or predselect
    autoScaleBatchSize = False
    findLR = False
    runMode = "TrainTest"    # "onlyTrain" # "onlyTest" # "TrainTest"
    changeAnyTesthparam = False  #True when wanting to change any Test hparams in a saved checkpiont when resuming TrainTest or onlyTest
  
    



    if MaskMergeMode == "sum":
        ChngdMaskMergeMode = "predselect"  
    elif MaskMergeMode == "predselect":
        ChngdMaskMergeMode = "sum"  

    if "Initial" in run_prefix:
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







    parser = ArgumentParser()
    parser = Classifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    hparams.out_act = out_act ####
    hparams.lr = lr ####
    hparams.w_d = w_d   #
    hparams.g_c_val = g_c_val   #
    hparams.g_c_algo = g_c_algo #
    hparams.max_epochs = num_epochs ####
    hparams.batch_size = batch_size ###
    hparams.trainID = trainID
    hparams.normmode = normmode
    hparams.percentile = percentile ####
    hparams.network = network
    hparams.model_segclassify = model_segclassify
    hparams.accumulate_grad_batches = accumulate_grad_batches   ####
    hparams.use_amp = use_amp   ####
    hparams.workers = workers   ####
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
    hparams.autoScaleBatchSize = autoScaleBatchSize ####
    hparams.findLR = findLR ####
    hparams.upalgo = upalgo
    hparams.dropout = dropout





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
        gpus=1,     # specifying number of gpus to be used = int    #alter is to use [] to indicate which gpu to use ex = [1] use gpu 1
        checkpoint_callback=True,  #checkpoint_callback=checkpoint_callback,   was for old trainer.py istead of 326, 327
        callbacks=[checkpoint_callback],
        max_epochs=hparams.max_epochs,
        terminate_on_nan=True,
        deterministic=True,
        accumulate_grad_batches=accumulate_grad_batches,
        resume_from_checkpoint=chkpoint,
        auto_scale_batch_size='binsearch' if autoScaleBatchSize else None,
        auto_lr_find=findLR,
        gradient_clip_val=g_c_val,
        gradient_clip_algorithm=g_c_algo
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
            raise RuntimeError("the next line will produce an error, resume_from_Checkpoint is set to False")
        model = Classifier.load_from_checkpoint(chkpoint)
        #If there us any change in Testhparam, specify it below
        if changeAnyTesthparam:
            model.hparams.MaskMergeMode = ChngdMaskMergeMode
            model.hparams.trainID = run_prefix + "_" + Dataset + "_" + amount + "_" + orient + "_" + network + "_" + upalgo + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
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
                raise RuntimeError("the next line will produce an error, resume_from_Checkpoint is set to False")
            model = Classifier.load_from_checkpoint(chkpoint)
            #specify the change in Test hParameters below
            model.hparams.MaskMergeMode = ChngdMaskMergeMode
            #specify the change "chng" in train id for chngd output folder name
            model.hparams.trainID = run_prefix + "_" + Dataset + "_" + amount + "_" + orient + "_" + network + "_" + upalgo + "_" + ("WClsW" if useClassWeight else "WoClsW") + \
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
