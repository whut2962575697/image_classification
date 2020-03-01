# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/22 16:04   xin      1.0         None
'''
from .wrappers import (
    SegmentationTTAWrapper,
    ClassificationTTAWrapper,
)
from .base import Compose

from .transforms import (
    HorizontalFlip, VerticalFlip, Rotate90, Scale, Add, Multiply, FiveCrops, Resize
)

from . import aliases