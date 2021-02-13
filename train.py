import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
import csv

from utils import parse_args, seed_reproducer

# Training settings
args = parse_args(sys.argv[1:])
use_cuda = torch.cuda.is_available()
seed_reproducer()

# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)
model_dir = os.path.join(args.experiment, 'checkpoints')
log_dir = os.path.join(args.experiment, 'logs')
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

from sklearn.model_selection import KFold
from data import BirdDataSetUnlabeled, BirdDataSetLabeled, return_data_transforms, mixup_data , return_data_test_transforms
from loss_function import alpha_weight, mixup_criterion, CenterLoss, LabelSmoothingCrossEntropy, linear_combination, reduce_loss
from model import Net, Net2

def validation():
    model.eval()
    correct = 0
    val_loss = 0
    for (data, target) in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output, _ = model(data)

        criterion = LabelSmoothingCrossEntropy(reduction='mean')
        val_loss += criterion(output, target).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    val_acc = 100. * correct / len(val_loader.dataset)
    val_loss /= len(val_loader.dataset)
    return val_acc, val_loss

def supervised_train():
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output, features = model(data)
        
        if args.mixup:
            data, target_a, target_b, lam = mixup_data(data, target,
                                                       args.alpha, use_cuda)
            data, target_a, targes_b = map(Variable, (data,
                                                      target_a, target_b))
       # alpha=0.005
        criterion = LabelSmoothingCrossEntropy(reduction='mean')
        if args.mixup:
            loss = mixup_criterion(criterion, output, target_a, target_a, lam)
        else:
            loss = criterion(output, target)
     #   loss = center_loss(features, target) * alpha + loss
        loss.backward()
        optimizer.step()
        scheduler.step()
     #   optimizer_centloss.zero_grad()
      #  for param in center_loss.parameters():
            # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
      #      param.grad.data *= (1./ alpha)
      #  optimizer_centloss.step()        
def semi_supervised_train():
    global step
    model.train()
    for batch_idx, data in enumerate(unlabeled_loader):
        if use_cuda:
            data = data.cuda()
        model.eval()
        output_unlabeled, _ = model(data)
        idx = output_unlabeled.softmax(1).max(1)[0] >= 0.9
        output_unlabeled = output_unlabeled[idx]
        if output_unlabeled.nelement() == 0:
                continue
        pseudo_label = output_unlabeled.data.max(1)[1]
        data = data[idx]

        model.train()
       # optimizer_centloss.zero_grad()
        optimizer.zero_grad()
       # alpha=0.005
        output, features = model(data)
        criterion = LabelSmoothingCrossEntropy(reduction='mean')
        unlabeled_loss = alpha_weight(step, T1, T2, af) * criterion(output, pseudo_label)
       # unlabeled_loss = center_loss(features, pseudo_label) * alpha + unlabeled_loss
        unlabeled_loss.backward()
        optimizer.step()
        scheduler.step()

T1 = args.T1
T2 = args.T2
af = args.af
step = 0

df = pd.read_csv(args.data_csv)
df_ext = pd.read_csv(args.external_data_csv)
train_idx = df['fold'] != args.k
test_idx = df['fold'] == args.k
train_dataset = df[train_idx]
test_dataset = df[test_idx]


data_transforms = return_data_transforms(args.input_size)
data_test_transforms = return_data_test_transforms(args.input_size)
unlabeled_loader = torch.utils.data.DataLoader(
    BirdDataSetUnlabeled(df_ext, transform=data_transforms, threshold=args.threshold),
    batch_size=args.batch_size, shuffle=True, num_workers=1)    
train_loader = torch.utils.data.DataLoader(
    BirdDataSetLabeled(train_dataset, transform=data_transforms, threshold=args.threshold),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    BirdDataSetLabeled(test_dataset, transform=data_test_transforms, threshold=args.threshold),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

# Model, optimizer and scheduler
if args.arch == 'efficientnet':
    model = Net()
else:
    model = Net2()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

best_val_acc = 0
start_epoch = 0
if len(args.checkpoint) > 0:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc'] 
    step = checkpoint['step']

if args.freeze == 1:
    for param in model.parameters():
        param.requires_grad = False
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
else:
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.lr, weight_decay=1e-5)


if (start_epoch-1) + args.epochs + 1 < T1:
    num_unlabeled_steps = 0
else:
    num_unlabeled_steps = (start_epoch-1) + args.epochs + 1 - max(T1, start_epoch)
num_labeled_steps = args.epochs


scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.max_lr, num_labeled_steps * len(train_loader) + len(unlabeled_loader)*  num_unlabeled_steps) 

if len(args.checkpoint) == 0:
    with open(log_dir+ '/logs_{}.csv'.format(args.k), 'w', newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            log_writer.writerow(["epoch", "val_acc", "val_loss"])
print('Training from epoch={}, until epoch={} on fold {}'.format(start_epoch, start_epoch + args.epochs-1, args.k))


#center_loss = CenterLoss(num_classes=20, feat_dim=model.num_ftrs, use_gpu=use_cuda)
#optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=0.001)

for epoch in range(start_epoch, start_epoch + args.epochs):
    if epoch < T1 or not args.semi_supervised:
        supervised_train()
    else:
        semi_supervised_train()
        supervised_train()
    step += 1
    val_acc, val_loss = validation()
    
    print("Epoch: {:02d} / {:02d} | Alpha Weight: {:.2f} | step: {:02d} | Val acc: {:.2f} | Val loss {:6f}".format(epoch, start_epoch + args.epochs -1, alpha_weight(step, T1, T2, af), step, val_acc, val_loss))        

    with open(log_dir + '/logs_{}.csv'.format(args.k), 'a', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow([epoch, val_acc.item(), val_loss])        

    if args.save_best_only and best_val_acc < val_acc:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'step': step,
            'input_size': args.input_size,
            'arch': args.arch,
            },  model_dir + '/checkpoints_{}_{}.pth'.format(args.k, epoch))
    if args.save_best_only == False:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'step': step,
            'input_size': args.input_size,
            'arch': args.arch
            },  model_dir + '/checkpoints_{}_{}.pth'.format(args.k, epoch))
