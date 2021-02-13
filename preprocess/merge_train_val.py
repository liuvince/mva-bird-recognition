import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='RecVis A3 Merge Data script')
parser.add_argument('--src', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--dst', type=str, default='bird_dataset/merged_images', metavar='D',
                    help="folder where data will be merged")

args = parser.parse_args()

train_folder = os.path.join(args.src, 'train_images')
val_folder = os.path.join(args.src, 'val_images')
labels = sorted(os.listdir(train_folder))

def copy_files_from_folder(src_folder, dst_folder):
  for f in os.listdir(src_folder):
    shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, f))

def make_dir(dir):
  if not os.path.exists(dir):
    os.mkdir(dir)

def make_multiple_subdirs(dir, subdirs):
  for subdir in subdirs:
    make_dir(os.path.join(dir, subdir))

make_dir(args.dst)
make_multiple_subdirs(args.dst, labels)

# Fill merged_folder with images
for label in labels:
  dst_folder = os.path.join(args.dst, label)
  
  copy_files_from_folder(os.path.join(train_folder, label), dst_folder)
  copy_files_from_folder(os.path.join(val_folder, label), dst_folder)
