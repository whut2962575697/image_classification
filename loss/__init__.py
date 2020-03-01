# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/21 10:17   xin      1.0         None
'''


import torch.nn as nn


class BaseLine_Loss(nn.modules.loss._Loss):
    def __init__(self, device):
        super(BaseLine_Loss, self).__init__()
        self.device = device

    def forward(self, outputs, targets):
        ce_loss = nn.CrossEntropyLoss().to(self.device)
        CE_Loss = ce_loss(outputs, targets)
        print('\r[loss] ce:%.2f\t ' % (CE_Loss.data.cpu().numpy(),), end=' ')
        return CE_Loss


def make_loss(cfg, device):
    if cfg.MODEL.NAME =='baseline':
        loss = BaseLine_Loss(device)
    return loss
