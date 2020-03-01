# -*- encoding: utf-8 -*-
'''
@File    :   baseline.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/21 13:38   xin      1.0         None
'''

import torch.nn as nn


from .backbone.resnet import resnet50, resnet34, resnet18, resnet101, resnet152
from .backbone.senet import seresnet50, seresnet101, seresnet152, senet154
from .backbone.gen_efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, tf_efficientnet_b3, \
    tf_efficientnet_b4, tf_efficientnet_b5
from .backbone.wrn import WideResNet

from .module.nonlocal_block import NONLocalBlock2D


class BaseLine(nn.Module):
    def __init__(self, n_channel=3, n_class=9, backbone='resnet50', dropout=0.5, use_nonlocal=False, use_scse=False, use_attention=False):
        super().__init__()
        self.n_cls = n_class
        self.inChannel = n_channel
        if backbone == "resnet50":
            self.base_model = resnet50(pretrained=True, in_chans=n_channel)
        elif backbone == "resnet34":
            self.base_model = resnet34(pretrained=True, in_chans=n_channel)
        elif backbone == "resnet18":
            self.base_model = resnet18(pretrained=True, in_chans=n_channel)
        elif backbone == "resnet101":
            self.base_model = resnet101(pretrained=True, in_chans=n_channel)
        elif backbone == "resnet152":
            self.base_model = resnet152(pretrained=True, in_chans=n_channel)
        elif backbone == "seresnet50":
            self.base_model = seresnet50(pretrained=True, in_chans=n_channel)
        elif backbone == "seresnet101":
            self.base_model = seresnet101(pretrained=True, in_chans=n_channel)
        elif backbone == "seresnet152":
            self.base_model = seresnet152(pretrained=True, in_chans=n_channel)
        elif backbone == "senet154":
            self.base_model = senet154(pretrained=True, in_chans=n_channel)
        elif backbone == "efficientnetb0":
            self.base_model = efficientnet_b0(pretrained=True, in_chans=n_channel)
        elif backbone == "efficientnetb1":
            self.base_model = efficientnet_b1(pretrained=True, in_chans=n_channel)
        elif backbone == "efficientnetb2":
            self.base_model = efficientnet_b2(pretrained=True, in_chans=n_channel)
        elif backbone == "efficientnetb3":
            self.base_model = efficientnet_b3(pretrained=True, in_chans=n_channel)
        elif backbone == "tf_efficientnetb3":
            self.base_model = tf_efficientnet_b3(pretrained=True, in_chans=n_channel)
        elif backbone == "efficientnetb4":
            self.base_model = efficientnet_b4(pretrained=True, in_chans=n_channel)
        elif backbone == "tf_efficientnetb4":
            self.base_model = tf_efficientnet_b4(pretrained=True, in_chans=n_channel)
        elif backbone == "efficientnetb5":
            self.base_model = efficientnet_b5(pretrained=True, in_chans=n_channel)
        elif backbone == "tf_efficientnetb5":
            self.base_model = tf_efficientnet_b5(pretrained=True, in_chans=n_channel)
        elif backbone == "efficientnetb6":
            self.base_model = efficientnet_b6(pretrained=True, in_chans=n_channel)
        elif backbone == "efficientnetb7":
            self.base_model = efficientnet_b7(pretrained=True, in_chans=n_channel)
        elif backbone == 'wrn40_4':
            self.base_model = WideResNet(40, n_class, 4, 0, use_nonlocal, use_scse, use_attention)
        elif backbone == 'wrn28_10':
            self.base_model = WideResNet(28, n_class, 10, 0, use_nonlocal, use_scse, use_attention)
        # elif backbone == 'mobilenet':
        #     self.base_model =
        else:
            self.base_model = None
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        if hasattr(self.base_model, 'fc'):
            num_ftrs = self.base_model.fc.in_features
        elif hasattr(self.base_model, 'classifier'):
            num_ftrs = self.base_model.classifier.in_features
        elif hasattr(self.base_model, 'last_linear'):
            num_ftrs = self.base_model.last_linear.in_features
        else:
            num_ftrs = None
        # self.reduce_layer = nn.Conv2d(num_ftrs*2, num_ftrs, 1)
        self.fc = nn.Linear(num_ftrs, n_class)

        # self.bnneck = nn.BatchNorm1d(num_ftrs)
        # self.bnneck.bias.requires_grad_(False)  # no shift

    def forward(self, x):
        feature = self.base_model.forward_features(x, False)

        # global_avg_feature = self.global_avg_pool(feature)
        # global_max_feature = self.global_max_pool(feature)
        # global_feature = torch.cat([global_avg_feature, global_max_feature], dim=1)
        # global_feature = self.reduce_layer(global_feature).view(feature.size(0), -1)
        global_feature = self.global_pool(feature).view(feature.size(0), -1)

        # global_feature = self.bnneck(global_feature)
        global_feature = self.dropout(global_feature)
        output = self.fc(global_feature)
        return output


