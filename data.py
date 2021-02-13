import zipfile
import os

import torch
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensor

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import random
from io import BytesIO

class BirdDataSetLabeled(Dataset):
    def __init__(self, datacsv, transform, threshold):
      self.filenames = datacsv['Id'].values
      self.targets = datacsv['Category'].values
      self.scores = datacsv['confidence'].values
      self.lefts = datacsv['left'].values
      self.tops = datacsv['top'].values
      self.rights = datacsv['right'].values
      self.bottoms = datacsv['bottom'].values
      self.transform = transform

      self.threshold = threshold
    def __getitem__(self, index):
      image = Image.open(self.filenames[index]).convert('RGB')

      confidence, left, top, right, bottom = self.scores[index], self.lefts[index], self.tops[index], self.rights[index], self.bottoms[index]
      if confidence >= self.threshold:
        W, H = image.size
        increase = random.randint(0, 3)
        # Crop with bounding box coordinates
        if increase == 0:
          image = image.crop((left, top, right, bottom))
        elif increase == 1:
          increase_rate = random.uniform(0, 0.2)
          image = image.crop(self.expand_box((left, top, right, bottom), increase_rate, H, W))
        # Keep Aspect Ratio
        elif increase == 2:
          cx, cy, cr = self.center_radius((left, top, right, bottom))
          image = image.crop(self.bbox(cx, cy, cr))
        else:
          increase_rate = random.uniform(0, 0.1)
          cx, cy, cr = self.center_radius((left, top, right, bottom))
          image = image.crop(self.bbox(cx, cy, cr + cr * increase_rate))       
      x = self.transform(image)
      y = self.targets[index]
      return x, y
    def __len__(self):
        return len(self.filenames)
    def expand_box(self, box, inc, H, W):
        (left, top, right, bottom) = box
        left = max(0, left - left * inc)
        right = min(W-1, right + right * inc)
        top = max(0, top - top * inc)
        bottom = min(H-1, bottom + bottom * inc)
        return (left, top, right, bottom)
    def center_radius(self, box):
        (left, top, right, bottom) = box
        x, y, w, h = left, top, right-left, bottom-top
        cx = x+w//2
        cy = y+h//2
        cr  = max(w,h)//2
        return cx, cy, cr
    def bbox(self, cx, cy, cr):
        left = cx - cr
        top = cy - cr 
        right = cx + cr
        bottom = cy + cr
        return (left, top, right, bottom)
class BirdDataSetUnlabeled(Dataset):
    def __init__(self, datacsv, transform, threshold):
      self.filenames = datacsv['Id'].values
      self.transform = transform
      self.scores = datacsv['confidence'].values
      self.lefts = datacsv['left'].values
      self.tops = datacsv['top'].values
      self.rights = datacsv['right'].values
      self.bottoms = datacsv['bottom'].values
      self.threshold = threshold
    def __getitem__(self, index):
      image = Image.open(self.filenames[index]).convert('RGB')

      confidence, left, top, right, bottom = self.scores[index], self.lefts[index], self.tops[index], self.rights[index], self.bottoms[index]
      if confidence >= self.threshold:
        W, H = image.size
        increase = random.randint(0, 3)
        # Crop with bounding box coordinates
        if increase == 0:
          image = image.crop((left, top, right, bottom))
        elif increase == 1:
          increase_rate = random.uniform(0, 0.2)
          image = image.crop(self.expand_box((left, top, right, bottom), increase_rate, H, W))
        # Keep Aspect Ratio
        elif increase == 2:
          cx, cy, cr = self.center_radius((left, top, right, bottom))
          image = image.crop(self.bbox(cx, cy, cr))
        else:
          increase_rate = random.uniform(0, 0.1)
          cx, cy, cr = self.center_radius((left, top, right, bottom))
          image = image.crop(self.bbox(cx, cy, cr + cr * increase_rate))       
      x = self.transform(image)
      return x
    def __len__(self):
        return len(self.filenames)

    def expand_box(self, box, inc, H, W):
        (left, top, right, bottom) = box
        left = max(0, left - left * inc)
        right = min(W, right + right * inc)
        top = max(0, top - top * inc)
        bottom = min(H, bottom + bottom * inc)
        return (left, top, right, bottom)

    def center_radius(self, box):
        (left, top, right, bottom) = box
        x, y, w, h = left, top, right-left, bottom-top
        cx = x+w//2
        cy = y+h//2
        cr  = max(w,h)//2
        return cx, cy, cr
    def bbox(self, x, cy, cr):
        left = cx - cr
        top = cy - cr 
        right = cx + cr
        bottom = cy + cr
        return (left, top, right, bottom)

def return_data_transforms(input_size):
    data_transforms = transforms.Compose([
        transforms.Resize((input_size+10, input_size+10)),
        transforms.ColorJitter(brightness=0.05, hue=.05, saturation=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((input_size, input_size)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.05)),
    ])
    return data_transforms

def return_data_test_transforms(input_size):
    data_transforms = transforms.Compose([
        transforms.Resize((input_size+10, input_size+10)),
        transforms.CenterCrop((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ])
    return data_transforms

# https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
