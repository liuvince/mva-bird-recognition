import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 20

from efficientnet_pytorch import EfficientNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.num_ftrs = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.classifier = nn.Linear(self.num_ftrs, nclasses)
    def forward(self, x):
        y = self.backbone.forward(x)
        x = self.classifier(y)
        return (x, y)

import pretrainedmodels
model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.backbone = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        self.backbone.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_ftrs = self.backbone.last_linear.in_features
        self.backbone.last_linear = nn.Identity()
        self.classifier = nn.Linear(self.num_ftrs, nclasses)
    def forward(self, x):
        y = self.backbone.forward(x)
        x = self.classifier(y)
        return (x, y)
