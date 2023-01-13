from __future__ import absolute_import

import torch
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from PIL import Image
import random
import math
import numpy as np
import numbers
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms.functional as TF
from typing import Sequence
# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Compose_MS(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask1, mask2, mask):
        for t in self.transforms:
            img, mask1, mask2, mask = t(img, mask1, mask2, mask)
        return img, mask1, mask2, mask


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomHorizontallyFlip_MS(object):
    def __call__(self, img, mask1, mask2, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask1.transpose(Image.FLIP_LEFT_RIGHT), mask2.transpose(
                Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask1, mask2, mask


class RandomResizedCrop(standard_transforms.RandomResizedCrop):
    def __call__(self, img, mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), F.resized_crop(mask, i, j, h, w,
                                                                                              self.size,
                                                                                              self.interpolation)

class RandomRotate:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles
    
    def __call__(self, img, mask):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle), TF.rotate(mask, angle)   

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
        # print('img')
        # print(img.size)
        # print('mask')
        # print(mask.size)
        assert img.size == mask.size
        w, h = img.size

        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            if w < tw:
              tw = w
            if h < th:
              th = h
            # return img.resize(tw, th, Image.BILINEAR, refcheck=False), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomCrop_MS(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask1, mask2, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            mask1 = ImageOps.expand(mask1, border=self.padding, fill=0)
            mask2 = ImageOps.expand(mask2, border=self.padding, fill=0)
        # print('img')
        # print(img.size)
        # print('mask')
        # print(mask.size)
        assert img.size == mask.size
        w, h = img.size

        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize(tw, th, Image.BILINEAR, refcheck=False), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        x1 = int(x1 / 4) * 4
        y1 = int(y1 / 4) * 4
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask1.crop(
            (int(x1 / 4), int(y1 / 4), int((x1 + tw) / 4), int((y1 + th) / 4))), mask2.crop(
            (int(x1 / 2), int(y1 / 2), int((x1 + tw) / 2), int((y1 + th) / 2))), mask.crop((x1, y1, x1 + tw, y1 + th))


# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor * self.para
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor == 1:
            return img
        tmp = np.array(img.resize((w // self.factor, h // self.factor), Image.BICUBIC)) * self.factor * self.factor
        img = Image.fromarray(tmp)
        return img


# def __call__(self, mask):
#     return mask.resize((self.size[1] / cfg.TRAIN.DOWNRATE, self.size[0] / cfg.TRAIN.DOWNRATE), Image.NEAREST)