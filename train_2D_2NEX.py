#run this to train the model

import argparse
import re
import os, datetime, time, glob
import numpy as np
import torch
import torch.nn as nn
from skimage.io import 
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau
import generator as dg
from generator import Dataset
from Model import Denoise2D,DoubleModule,DoubleM
import ssim

# Params
parser = argparse.ArgumentParser(description='PyTorch MriData')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--epoch', default=500, type=int, help='max number of train epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--data_dir', default='./train', type=str, help='directory of dataset')
parser.add_argument('--full_folder', default = ['dataset1'], type=str, help='list of input files')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
data_dir = args.data_dir
batch_size = args.batch_size
n_epoch = args.epoch
full_folder = args.full_folder
save_dir = os.path.join('./', args.model)# type the model saving folder here

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def normalize(x):
    x = x.astype('float32')
    x = torch.from_numpy(x.transpose(0,3,1,2))
    return x

def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%_1M:%S:"), *args, **kwargs)

if __name__ == '__main__':
    print('===> Building model')
    model = Denoise2D()
    initial_epoch = findLastCheckpoint(save_dir=save_dir) 
    if initial_epoch > 0:
        print('resuming by loading epoch %04d' % initial_epoch)
        model = torch.load(os.path.join(save_dir, 'model_%04d.pth' % initial_epoch))

    criterion = torch.nn.MSELoss()
    cs = ssim.ssim
    
    if cuda:
        model.to(device)
        criterion = criterion.to(device)
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer,'min',factor=0.2,patience=10,verbose=False,threshold=1e-3)
    
    x,y,z = dg.datagenerator(data_dir=data_dir,folder=full_folder)
    x = normalize(x)
    y = normalize(y)
    z = normalize(z)
    Dataset=Dataset(x,y,z)
        
    optimizer.step() 
    epoch_count = 0
    for epoch in range(initial_epoch, n_epoch):
        epoch_count += 1
        DLoader=DataLoader(dataset=Dataset, num_workers=12, drop_last=True, batch_size=batch_size, shuffle=True)
        
        epoch_loss = 0
        start_time = time.time()
        
        for n_count, batch_yx in enumerate(DLoader): 
            if cuda:
                batch_x, batch_y, batch_m= batch_yx[0].to(device), batch_yx[1].to(device),batch_yx[2].to(device)
            optimizer.zero_grad()
            batch = model(batch_x,batch_y)
            loss = criterion(batch,batch_m)*(1-cs(batch,batch_m))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if n_count % 200 == 0:
                print('%4d %4d / %4d loss = %4.4f ' % (epoch+1, n_count, x.size(0)//batch_size, loss.item()/batch_size))
        
        scheduler.step(epoch_loss)
        elapsed_time = time.time() - start_time
        log('epoch = %4d, loss = %4.4f, time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
        
        torch.save(model, os.path.join(save_dir, 'model_%04d.pth' % (epoch+1)))