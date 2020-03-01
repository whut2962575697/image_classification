# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/21 10:18   xin      1.0         None
'''

from .baseline import BaseLine


def build_model(cfg):
    if cfg.MODEL.NAME == "baseline":
        model = BaseLine(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.BACKBONE, cfg.MODEL.DROPOUT, cfg.MODEL.USE_NONLOCAL, cfg.MODEL.USE_SCSE, cfg.MODEL.USE_ATTENTION)
    return model

