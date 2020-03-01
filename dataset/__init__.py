# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/21 10:17   xin      1.0         None
'''


import torchvision.transforms as T
import albumentations as A
from torch.utils.data.dataloader import DataLoader
from .custom import Augmentation, RandomErasing



from .data import Fashion_MNIST_DataSet
from common.autoaugment.achive import autoaug_policy


def get_trm(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        main_transform = A.Compose([
            A.Resize(cfg.INPUT.RESIZE_TRAIN[0], cfg.INPUT.RESIZE_TRAIN[1]),
            A.RandomCrop(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]),

            A.HorizontalFlip(p=cfg.INPUT.PROB),
            # A.RandomRotate90(p=1),
            # A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=.1, rotate_limit=10),
            # A.OneOf([A.RandomContrast(limit=0.2, p=0.1),
            #        A.RandomGamma(gamma_limit=(90, 110), p=0.5),
            #        A.RandomBrightness(limit=0.2, p=0.2)], p=0.5),
            # A.CoarseDropout()
        ], p=1)

        if cfg.INPUT.USE_AUTOAUG:
            image_transform_list = [
                Augmentation(autoaug_policy()),

                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RANDOM_ERASE.RE_PROB, sh=cfg.INPUT.RANDOM_ERASE.RE_MAX_RATIO,
                              mean=cfg.INPUT.PIXEL_MEAN),
            ]
        else:
            image_transform_list = [

                T.ToTensor(),
                normalize_transform,
                RandomErasing(probability=cfg.INPUT.RANDOM_ERASE.RE_PROB, sh=cfg.INPUT.RANDOM_ERASE.RE_MAX_RATIO,
                              mean=cfg.INPUT.PIXEL_MEAN),
            ]

        image_transform = T.Compose(image_transform_list)

    else:
        main_transform = A.Compose\
                ([
            A.Resize(cfg.INPUT.RESIZE_TEST[0], cfg.INPUT.RESIZE_TEST[1]),
            A.CenterCrop(cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])
        ])
        image_transform = T.Compose([
            T.ToTensor(),
            normalize_transform,
        ])

    return main_transform, image_transform


def make_dataloader(cfg, num_gpus):
    train_main_transform, train_image_transform = get_trm(cfg)
    val_main_transform, val_image_transform = get_trm(cfg, False)
    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
    train_dataset = Fashion_MNIST_DataSet(cfg, mode='train',  main_transform=train_main_transform,
                                     img_transform=train_image_transform)
    val_dataset = Fashion_MNIST_DataSet(cfg, mode='val', main_transform=val_main_transform,
                                   img_transform=val_image_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader

