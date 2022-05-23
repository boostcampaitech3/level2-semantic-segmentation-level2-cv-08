import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.optim.lr_scheduler as lr_scheduler

def get_loss(loss):
    if loss == 'CE':
        return nn.CrossEntropyLoss()
    elif loss == 'focal':
        return smp.losses.FocalLoss('multiclass')
    elif loss == 'softCE':
        return smp.losses.SoftCrossEntropyLoss('multiclass')

def get_scheduler(scheduler, optimizer):
    if scheduler == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)
    else:
        sc = getattr(lr_scheduler, scheduler)
        return sc(optimizer)