# -*- encoding: utf-8 -*-
'''
@File    :   data.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/21 12:45   xin      1.0         None
'''


from torch.utils import data
import os
import torch
import random
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os.path as osp
import numpy as np
import torchvision

from utils.augmix.augmix import augment_and_mix


def read_image(img_path, img_type='RGB'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            if img_type == 'RGB':
                img = Image.open(img_path).convert(img_type)
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. "
                            "Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataImageSet(data.Dataset):
    def __init__(self, cfg, mode, image_type, main_transform, image_transform):
        self.mode = mode
        self.cfg = cfg
        self.image_type = image_type
        self.main_transform = main_transform
        self.image_transform = image_transform
        self.mode = mode
        if self.mode == 'train':
            self.file_list = [x for x in
                              os.listdir(os.path.join(cfg.DATASETS.DATA_PATH, cfg.DATASETS.TRAIN.IMAGE_FOLDER))
                              if x.endswith(image_type)]
        elif self.mode == 'val':
            self.file_list = [x for x in
                              os.listdir(os.path.join(cfg.DATASETS.DATA_PATH, cfg.DATASETS.VAL.IMAGE_FOLDER))
                              if x.endswith(image_type)]

        self.num_samples = len(self.file_list)

    def __getitem__(self, index):
        data, gt = self.read_data_and_gt(index)
        return data, gt

    def __len__(self):
        return self.num_samples

    def read_data_and_gt(self, index):
        if self.mode == 'train':
            img = read_image(os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.TRAIN.IMAGE_FOLDER, self.file_list[index]))
            s = os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.TRAIN.IMAGE_FOLDER, self.file_list[index])
        elif self.mode == 'val':
            s = os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.VAL.IMAGE_FOLDER, self.file_list[index])
            img = read_image(
                os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.VAL.IMAGE_FOLDER, self.file_list[index]))
        img = self.main_transform(img)
        img = self.image_transform(img)
        if self.mode == 'train':
            s1 = os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.TRAIN.GT_FOLDER, self.file_list[index])
            gt = read_image(os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.TRAIN.GT_FOLDER, self.file_list[index]), None)
        elif self.mode == 'val':
            gt = read_image(
                os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.VAL.GT_FOLDER, self.file_list[index]),
                None)
        gt = self.main_transform(gt)
        gt = torch.LongTensor(np.array(gt))

        return img, gt

    def get_num_samples(self):
        return self.num_samples


class Fashion_MNIST_DataSet(data.Dataset):
    def __init__(self, cfg, mode, main_transform, img_transform):
        self.mode = mode
        self.cfg = cfg

        self.main_transform = main_transform
        self.img_transform = img_transform

        self.mode = mode
        if self.mode == 'train':
            self.dataset = torchvision.datasets.FashionMNIST(root=cfg.DATASETS.DATA_PATH, train=True, download=True,
                                                              )

        elif self.mode == 'val':
            self.dataset = torchvision.datasets.FashionMNIST(root=cfg.DATASETS.DATA_PATH, train=False, download=True,
                                                            )

        self.num_samples = len(self.dataset)

    def __getitem__(self, index):
        data, label = self.read_data_and_gt(index)
        return data, label

    def __len__(self):
        return self.num_samples

    def read_data_and_gt(self, index):
        img, label = self.dataset[index]
        img = np.array(img)
        img = np.expand_dims(img,2).repeat(3, axis=2)
        aug = self.main_transform(image=img)
        img = aug['image']

        if self.cfg.INPUT.USE_AUGMIX and self.mode == 'train' and np.random.uniform()>0.5:
            img = augment_and_mix(Image.fromarray(img), self.img_transform)
        else:
            img = self.img_transform(Image.fromarray(img))

        # img = self.img_transform(Image.fromarray(img))

        label = torch.tensor(label).long()

        return img, label

    def get_num_samples(self):
        return self.num_samples

