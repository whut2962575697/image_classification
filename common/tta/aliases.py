# -*- encoding: utf-8 -*-
'''
@File    :   aliases.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/10/22 16:06   xin      1.0         None
'''

from .base import Compose, Merger
from . import transforms as tta



def flip_transform():
    return Compose([tta.HorizontalFlip(), tta.VerticalFlip()])


def hflip_transform():
    return Compose([tta.HorizontalFlip()])


def vlip_transform():
    return Compose([tta.VerticalFlip()])


def d4_transform():
    return Compose(
        [
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )

def multiscale_transform(scales, interpolation="nearest"):
    return Compose([tta.Scale(scales, interpolation=interpolation)])


def five_crop_transform(crop_height, crop_width):
    return Compose([tta.FiveCrops(crop_height, crop_width)])


def ten_crop_transform(crop_height, crop_width):
    return Compose([tta.HorizontalFlip(), tta.FiveCrops(crop_height, crop_width)])


def twenty_crop_transform(crop_height, crop_width):
    return Compose([tta.HorizontalFlip(), tta.VerticalFlip(), tta.FiveCrops(crop_height, crop_width)])


def custom_transform(model,crop_height, crop_width, image):
    labels1 = list()
    labels2 = list()
    labels3 = list()
    labels4 = list()
    for transformer in ten_crop_transform(crop_height, crop_width):  # custom transforms or e.g. tta.aliases.d4_transform()

        # augment image
        augmented_image = transformer.augment_image(image)

        # pass to model
        model_output1, model_output2, model_output3, model_output4  = model(augmented_image)

        # reverse augmentation for mask and label

        deaug_label1 = transformer.deaugment_label(model_output1)
        deaug_label2 = transformer.deaugment_label(model_output2)
        deaug_label3 = transformer.deaugment_label(model_output3)
        deaug_label4 = transformer.deaugment_label(model_output4)

        # save results

        labels1.append(deaug_label1)
        labels2.append(deaug_label2)
        labels3.append(deaug_label3)
        labels4.append(deaug_label4)
    mean_merge1 = Merger('mean', len(labels1))
    # reduce results as you want, e.g mean/max/min
    for label in labels1:
        mean_merge1.append(label)
    label1 = mean_merge1.result

    mean_merge2 = Merger('mean', len(labels2))
    # reduce results as you want, e.g mean/max/min
    for label in labels2:
        mean_merge2.append(label)
    label2 = mean_merge2.result

    mean_merge3 = Merger('mean', len(labels3))
    # reduce results as you want, e.g mean/max/min
    for label in labels3:
        mean_merge3.append(label)
    label3 = mean_merge3.result

    mean_merge4 = Merger('mean', len(labels4))
    # reduce results as you want, e.g mean/max/min
    for label in labels4:
        mean_merge4.append(label)
    label4 = mean_merge4.result

    return label1, label2, label3, label4


