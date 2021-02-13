import pandas as pd
from PIL import Image
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='RecVis A3 cropping image script')
parser.add_argument('--src', type=str, metavar='S',
                    help="folder where data is located")
parser.add_argument('--threshold', type=float, metavar='D',
                    help="threshold")
parser.add_argument('--inc', type=float, metavar='D',
                    help="threshold")
parser.add_argument('--square', type=int, default=1, metavar='D',
                    help="")
args = parser.parse_args()

def expand_box(box, inc, H, W):
  (left, top, right, bottom) = box
  left = max(0, left - left * inc)
  right = min(W-1, right + right * inc)
  top = max(0, top - top * inc)
  bottom = min(H-1, bottom + bottom * inc)
  return (left, top, right, bottom)

def area(box):
    (left, top, right, bottom) = box
    x, y, w, h = left, top, right-left, bottom-top
    return w * h

def center_radius(box):
    (left, top, right, bottom) = box
    x, y, w, h = left, top, right-left, bottom-top
    cx = x+w//2
    cy = y+h//2
    cr  = max(w,h)//2
    return cx, cy, cr
def bbox(cx, cy, cr):
    left = cx - cr
    top = cy - cr 
    right = cx + cr
    bottom = cy + cr
    return (left, top, right, bottom)

df = pd.read_csv(args.src)
filenames = df.loc[:, 'Id'].values
confidences = df.loc[:, 'confidence'].values
lefts = df.loc[:, 'left'].values
tops = df.loc[:, 'top'].values
rights = df.loc[:, 'right'].values
bottoms = df.loc[:, 'bottom'].values

for index, row in df.iterrows():
  filename = row['Id']
  confidence = row['confidence']
  left = row['left']
  top = row['top']
  right = row['right']
  bottom = row['bottom']
  if confidence != -1 and confidence > args.threshold:
    box = (left, top, right, bottom)
    image = Image.open(filename).convert('RGB')
    W, H = image.size
    if area(box) < 3600:
        cx, cy, cr = center_radius(box)
        r = cr * np.sqrt(6000/cr**2)
        image.crop(bbox(cx, cy, r)).save(filename,  quality=100)
    else:
        if args.square:
          cx, cy, cr = center_radius(box)
          image.crop(bbox(cx, cy, cr + args.inc*cr)).save(filename,  quality=100)
        else:
          image.crop(expand_box(box, args.inc, H, W)).save(filename,  quality=100)
