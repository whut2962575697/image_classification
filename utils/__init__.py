# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/21 10:18   xin      1.0         None
'''

from .metrics import AvgerageMeter
from .logging import setup_logger
from .mixup import mixup_data
from .cutmix import rand_bbox
from common.optimizer.ranger import Ranger


from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np
import torch.nn as nn
import random
import os


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_optimizer(model, opt, lr, weight_decay, momentum=0.9, nesterov=True):
    if opt == 'SGD':
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
    elif opt == 'AMSGRAD':
        optimizer = getattr(torch.optim,'Adam')(model.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
    elif opt == 'Ranger':
        optimizer = Ranger(params=filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr)
        # optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=lr, amsgrad=True)
    else:
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay)
    return optimizer


def calculate_score(cfg, outputs, targets):
    if cfg.MODEL.NAME == 'baseline':
        targets = targets.data.cpu()
        outputs = outputs.data.cpu()

        f1 = f1_score(
            targets,
            np.argmax(outputs, 1),
            average="macro")
        acc = accuracy_score(np.argmax(outputs, 1), targets)
    return f1, acc

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)