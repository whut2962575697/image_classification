import numpy as np
import random
import math

from common.autoaugment.augmentations import apply_augment

# class RandomErasing:
#     def __init__(self, p, area_ratio_range, min_aspect_ratio, max_attempt):
#         self.p = p
#         self.max_attempt = max_attempt
#         self.sl, self.sh = area_ratio_range
#         self.rl, self.rh = min_aspect_ratio, 1. / min_aspect_ratio
#
#     def __call__(self, image):
#         image = np.asarray(image).copy()
#
#         if np.random.random() > self.p:
#             return image
#
#         h, w = image.shape[:2]
#         image_area = h * w
#
#         for _ in range(self.max_attempt):
#             mask_area = np.random.uniform(self.sl, self.sh) * image_area
#             aspect_ratio = np.random.uniform(self.rl, self.rh)
#             mask_h = int(np.sqrt(mask_area * aspect_ratio))
#             mask_w = int(np.sqrt(mask_area / aspect_ratio))
#
#             if mask_w < w and mask_h < h:
#                 x0 = np.random.randint(0, w - mask_w)
#                 y0 = np.random.randint(0, h - mask_h)
#                 x1 = x0 + mask_w
#                 y1 = y0 + mask_h
#                 image[y0:y1, x0:x1] = np.random.uniform(0, 1)
#                 break
#
#         return image


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img