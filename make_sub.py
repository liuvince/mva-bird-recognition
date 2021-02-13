import os
import numpy as np
import pandas as pd
from scipy.special import softmax
import argparse

parser = argparse.ArgumentParser(description='Make submission')
parser.add_argument('--src', type=str, default="experiment/preds", metavar='D',
                    help="name of the directory with predictions")
parser.add_argument('--outfile', type=str, default='drive/My Drive/recvis/A3/sub.csv', metavar='D',
                    help="name of the directory to put predictions")
args = parser.parse_args()

src = args.src
outfile = args.outfile

avg_pred = np.zeros((517, 20))
for f in os.listdir(src):
  df = pd.read_csv(os.path.join(src, f))
  df = df.sort_values(by='id')
  avg_pred += softmax(df.iloc[:,1:].values, axis=0)
avg_pred /= len(os.listdir(src))

output_csv = pd.DataFrame(columns=['Id', 'Category'])
output_csv['Id'] = df.iloc[:, 0]
output_csv['Category'] = avg_pred.argmax(axis=1)
output_csv.to_csv(outfile, index=None)
