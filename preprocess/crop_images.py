import argparse
import torchvision
from torchvision import transforms
import torch
import os 
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps

parser = argparse.ArgumentParser(description='RecVis A3 cropping image script')
parser.add_argument('--src', type=str, metavar='S',
                    help="folder where data is located")
parser.add_argument('--dst', type=str, metavar='D',
                    help="folder where data will be cropped")
parser.add_argument('--external', type=int, default=0, metavar='E',
                    help="is bird_dataset or external")
parser.add_argument('--maskrcnn', type=int, default=1, metavar='E',
                    help="")
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
if args.maskrcnn:
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
else:
  model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
if use_cuda:
  model.cuda()
model.eval()

filenames = []

def crop_images_from_dir(src_image, threshold=0.4):
  x = Image.open(src_image).convert('RGB')
  x = transforms.ToTensor()(x).unsqueeze_(0)
  data = x.cuda()
  predictions = model(data)

  prediction = predictions[0]
  boxes = prediction['boxes'].cpu().detach().numpy().astype(int)
  labels = prediction['labels'].cpu().detach().numpy()
  scores = prediction['scores'].cpu().detach().numpy()
  idx_bird = (labels == 16)
  if idx_bird.sum() == 0:
    boxes = []
    scores = -1
  else:
    boxes = boxes[idx_bird][0]
    scores = scores[idx_bird][0]
  return src_image, boxes, scores

src_subdirs = []
if args.external:
  src_subdirs.append(args.src)
else:
  for d in ['merged']:
    src_dir = os.path.join(args.src, d)
    for subdir in os.listdir(src_dir):
      src_subdirs.append(os.path.join(src_dir, subdir))

Ids = []
Left = []
Top = []
Right = []
Bottom = []
Scores = []
for src_subdir in src_subdirs:
  for impath in os.listdir(src_subdir):
    ids, boxes_output, scores_output = crop_images_from_dir(os.path.join(src_subdir, impath))
    if scores_output == -1:
      (left, top, right, bottom) = (0,0,0,0)
    else:
      (left, top, right, bottom) = boxes_output
    Ids.append(ids)
    Scores.append(scores_output)
    Left.append(left)
    Top.append(top)
    Right.append(right)
    Bottom.append(bottom)
df = pd.DataFrame({'Id': Ids, 'confidence': Scores, 'left': Left, 'top': Top, 'right': Right, 'bottom': Bottom}).to_csv(args.dst, index=None)






