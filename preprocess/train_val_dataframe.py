import os
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description='RecVis A3 create data dataframe')
parser.add_argument('--src', type=str, metavar='S',
                    help="folder where data is located")
parser.add_argument('--dst', type=str, metavar='D',
                    help="where to save dataframe")

args = parser.parse_args()

def load_to_dataframe(merged):
    labels = sorted(os.listdir(merged))
    filenames = []
    targets = []
    for idx, label in enumerate(labels):
        directory = os.path.join(merged, label)
        filenames_add = [os.path.join(directory, f) for f in os.listdir(directory)]
        targets_add = [idx for _ in range(len(filenames_add))]

        filenames = filenames + filenames_add
        targets = targets + targets_add
    return pd.DataFrame({'Id': filenames, 'Category': targets})

df = load_to_dataframe(args.src)
skf = StratifiedKFold(n_splits=10, random_state=22, shuffle=True)
df['fold'] = 0
for fold, (train_index, test_index) in enumerate(skf.split(df.loc[:, 'Id'].values, df.loc[:, 'Category'].values)):
    df.loc[test_index, 'fold'] = fold

df.to_csv(args.dst, index=None)
