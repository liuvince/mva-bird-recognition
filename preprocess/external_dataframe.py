import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='RecVis A3 create data dataframe')
parser.add_argument('--src', type=str, metavar='S',
                    help="folder where data is located")
parser.add_argument('--dst', type=str, metavar='D',
                    help="where to save dataframe")

args = parser.parse_args()

filenames = [os.path.join(args.src, f) for f in os.listdir(args.src)]
pd.DataFrame({'Id': filenames}).to_csv(args.dst, index=None)
