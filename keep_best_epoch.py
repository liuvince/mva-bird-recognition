import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Keep best epoch')
parser.add_argument('--outfile', type=str, default='experiment/best_models.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--folds', type=int, default=1,
                    help="Num")
parser.add_argument('--avg', type=int, default=15,
                    help="Average how many epochs")
args = parser.parse_args()


best_fold = []
best_epoch = []
performances = []

for fold in range(0, args.folds):
  df = pd.read_csv("experiment/logs/logs_{}.csv".format(fold))
  df = df.drop_duplicates(subset=['epoch'], keep='last')
  df.index = df.reset_index()['index'].values
  for e in np.argsort(-df["val_acc"].values)[:args.avg]: 
    best_fold.append(fold)
    best_epoch.append(e)
    performances.append(df["val_acc"].values[e])

df_output = pd.DataFrame({'fold': best_fold,
                          'epoch': best_epoch,
                          'val_acc': performances})

df_output.to_csv(args.outfile, index=None)
