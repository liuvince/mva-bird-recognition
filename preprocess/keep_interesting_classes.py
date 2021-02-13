import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

keep_idx = np.array([416, # Brewer's Blackbird
    268, # Red-winged Blackbird
    404, # Rusty Blackbird
    911, # Rusty Blackbird
    306, # Yellow-header Blackbird
    254, # Bobolink
    706, # Indigo Bunting
    687, # Lazuli Bunting
    723, # Painted Bunting
    851, # Gray Catbird
    886, # Yellow-breasted Chat
    267, # Eastern Towhee
    75, # Brandt's Cormorant
    474, # Brandt's Cormorant
    590, # Bronzed Cowbird
    915, # Bronzed Cowbird
    435, # Brown Creeper
    827, # Brown Creeper
    857, # American Crow
    710, # Fish Crow
    139, # Black-billed Cuckoo
    534, # Black-billet Cuckoo
    533, # Yellow-billed Cuckoo
    77, # Yellow-billet Cuckoo
    255, #Gray-crowned Rosy-Finch
    917] # Gray-crowned Rosy-Finch
)
wd = "nabirds" # working dir
df = pd.read_csv(os.path.join(wd, "image_class_labels.txt"), sep = " ", header=None)
keep_bool = df.iloc[:, 1].apply(lambda row: row in keep_idx)
keep_images = df[keep_bool].iloc[:,0].values

df_filepath = pd.read_csv(os.path.join(wd, "images.txt"), sep = " ", header=None)
keep_paths = df_filepath.iloc[:, 0].apply(lambda row: row in keep_images)

dst_dir = "external_bird_dataset"
src_dir = os.path.join(wd, 'images')

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)
for i, filename in tqdm(enumerate(df_filepath[keep_paths].iloc[:, 1].values)):
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, '{}.jpeg'.format(i))
    shutil.copy(src, dst)

