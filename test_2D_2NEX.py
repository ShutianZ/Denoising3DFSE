# run this to test the model

import argparse
import os, time, datetime
import numpy as np
import torch.nn as nn
import torch
import cv2
import fnmatch
import SimpleITK as sitk
import pydicom
from Model import Denoise2D,DoubleModule,DoubleM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='./test', type=str, help='directory of test dataset')
    parser.add_argument('--model_dir', default='./models', type = str, help='directory of the model')
    parser.add_argument('--model_name', default='model.pth', type = str, help='the model name')
    parser.add_argument('--result_dir', default='./results', type=str, help='directory of the denoised result saving')
    parser.add_argument('--device_ids', default=0, type = int, help='gpu number')
    return parser.parse_args()


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def operate(path,verbose):
    if verbose:
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        im = np.squeeze(img)
    else:
        img = pydicom.dcmread(path)
        im = img.pixel_array
    im = np.array(im,dtype = np.float32)
    return im

def normal(x):
    x = torch.from_numpy(x)
    x = torch.squeeze(x)
    x = torch.unsqueeze(torch.unsqueeze(x,0),0)
    x = x.cuda(args.device_ids)
    return x

def normalize(x,WC,RI,RS):
    RescaleIntercept = RI
    RescaleSlope = RS
    window_center = WC
    window_width = WC*2
    normed_x = (x-RescaleIntercept)/RescaleSlope
    return normed_x
    
if __name__ == '__main__':

    args = parse_args()
    cuda = torch.cuda.is_available()
    device_ids = args.device_ids
    this_model_name =args.model_name

    if not os.path.exists(os.path.join(args.model_dir,this_model_name)):
        log('no such model_'+this_model_name)
        model = torch.load(os.path.join(args.model_dir, this_model_name))
    else:
        model = torch.load(os.path.join(args.model_dir, this_model_name))
        log('load trained model:'+this_model_name)

    if torch.cuda.is_available():
        model = model.cuda(args.device_ids)

    if not os.path.exists(os.path.join(args.result_dir)):
        os.mkdir(os.path.join(args.result_dir))

    with torch.no_grad():

        files = os.listdir(os.path.join(args.set_dir, 'magnitude'))
            
        files_1r = os.listdir(os.path.join(args.set_dir, 'NEX1r'))
        files_1i = os.listdir(os.path.join(args.set_dir, 'NEX1i'))
        files_2r = os.listdir(os.path.join(args.set_dir, 'NEX2r'))
        files_2i = os.listdir(os.path.join(args.set_dir, 'NEX2i'))
        
        i = 0
        start_time = time.time()
        for path in files:
            i = i + 1
            img = pydicom.dcmread(os.path.join(args.set_dir, path))
            WC = np.max(np.array(img.pixel_array,dtype = np.float32))/2
            RI = img.data_element('RescaleIntercept').value
            RS = img.data_element('RescaleSlope').value
            
            img_1r = operate(os.path.join(args.set_dir, files_1r[files.index(path)]),True)
            img_2r = operate(os.path.join(args.set_dir, files_2r[files.index(path)]),True)
            img_1i = operate(os.path.join(args.set_dir, files_1i[files.index(path)]),True)
            img_2i = operate(os.path.join(args.set_dir, files_2i[files.index(path)]),True)
            
            y1r = normal(img_1r)
            y1i = normal(img_1i)
            y2r = normal(img_2r)
            y2i = normal(img_2i)

            a = torch.cat((y1r,y1i),1)
            b = torch.cat((y2r,y2i),1)

            x_ = model(a,b)
            
            x_1 = torch.squeeze(torch.chunk(x_,2,1)[0])
            x_2 = torch.squeeze(torch.chunk(x_,2,1)[1])
            
            x = torch.sqrt(x_1*x_1 + x_2*x_2)
            x = normalize(x,WC,RI,RS)
            x = x.cpu().detach().numpy().astype(np.int16)

            img.PixelData = x.tobytes()
            img.save_as(os.path.join(args.result_dir, path))
        elapsed_time = time.time() - start_time
        print('%2.4f second' % (elapsed_time))
            