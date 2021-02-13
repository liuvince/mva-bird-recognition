import random
import os
import numpy as np
import torch
import argparse

def seed_reproducer(seed=2020):
    """Reproducer for pytorch experiment.
    https://github.com/alipay/cvpr2020-plant-pathology/blob/master/utils.py
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

def parse_args(args):
    parser = argparse.ArgumentParser(description='RecVis A3 training one fold script')
    parser.add_argument('--data_csv', type=str, metavar='D1',
		            help="folder where data_csv is located")
    parser.add_argument('--external_data_csv', type=str, default="", metavar='D2',
		            help="folder where external_data_csv is located")

    parser.add_argument('--batch_size', type=int, default=32, metavar='B',
		            help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
		            help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
		            help='learning rate (default: 0.001)')
    parser.add_argument('--max_lr', type=float, default=0.01, metavar='LR',
		            help='max learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
		            help='SGD momentum (default: 0.5)')
    parser.add_argument('--semi_supervised', type=int, default=1, metavar='K',
		            help="semi supervised or full supervised")

    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
		            help='folder where experiment outputs are located.')
    parser.add_argument('--save_best_only', type=int, default=0, metavar='K',
		            help="save only best model")
    parser.add_argument('--checkpoint', type=str, default="", metavar='K',
		            help="resume from checkpoint")
    parser.add_argument('--input_size', type=int, default=300, metavar='K',
		            help="input_size")
    parser.add_argument('--k', type=int, default=1, metavar='K',
		            help="number of fold for cross validation")
    parser.add_argument('--T1', type=int, default=60, metavar='K',
		            help="T1 Pseudo Label")
    parser.add_argument('--T2', type=int, default=320, metavar='K',
		            help="T2 Pseudo Label")
    parser.add_argument('--af', type=int, default=3, metavar='K',
		            help="af Pseudo Label")
    parser.add_argument('--freeze', type=int, default=1, metavar='K',
		            help="Freeze")
    parser.add_argument('--mixup', type=int, default=0, metavar='K',
		            help="mixup")
    parser.add_argument('--alpha', type=float, default=0.5, metavar='K',
		            help="mixup")
    parser.add_argument('--arch', type=str, default="efficientnet", metavar='A',
		            help="architecture")
    parser.add_argument('--threshold', type=float, default=0.6, metavar='K',
		            help="threshold")
    return parser.parse_args(args)
