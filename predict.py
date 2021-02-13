import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch

from model import Net, Net2

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--cropped', type=int, default=1, metavar='K',
		            help="cropped or not")
args = parser.parse_args()
use_cuda = torch.cuda.is_available()


state_dict = torch.load(args.model)
if state_dict['arch'] == 'efficientnet':
    model = Net()
else:
    model = Net2()
model.load_state_dict(state_dict['model_state_dict'])
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

from data import return_data_test_transforms 

data_transforms = return_data_test_transforms(state_dict['input_size'])

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

output_file = open(args.outfile, "w")
output_file.write('id,{}\n'.format(','.join([str(i) for i in range(20)])))
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()

        output,_ = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        pred_csv_row = "{}".format(f[:-4])
        for out in output.data[0]:
            pred_csv_row += ",{}".format(out.item())
        pred_csv_row += "\n"
        output_file.write(pred_csv_row)

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
