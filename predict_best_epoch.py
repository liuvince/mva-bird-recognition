import os
from tqdm import tqdm
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Predict best epoch')
parser.add_argument('--preds_dir', type=str, default='experiment/preds/', metavar='D',
                    help="name of the directory to put predictions")
parser.add_argument('--filename', type=str, default='experiment/best_models.csv', metavar='D',
                    help="name of the directory to put predictions")
parser.add_argument('--checkpoints', type=str, default='experiment/checkpoints', metavar='D',
                    help="name of the directory where checkpoint is located")
args = parser.parse_args()

df = pd.read_csv(args.filename)
checkpoint = args.checkpoints + '/checkpoints_{}_{}.pth'
preds_dir = args.preds_dir
if not os.path.exists(preds_dir):
  os.mkdir(preds_dir)

for index, row in tqdm(df.iterrows()):
  pred_outfile = os.path.join(preds_dir, 'preds_{:d}_{:d}'.format(int(row['fold']), int(row['epoch'])))
  model_outfile = checkpoint.format(int(row['fold']), int(row['epoch']))
  os.system('python3 code/predict.py --data data/bird_dataset --outfile {} --model {}'.format(pred_outfile, model_outfile))
