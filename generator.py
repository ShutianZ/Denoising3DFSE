import os
import torch
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset

patch_size, stride = 224,112

class Dataset(Dataset):

    def __init__(self, x, y ,z):
        super(Dataset, self).__init__()
        self.x = x
        self.y = y
        self.z = z
        
    def __getitem__(self, index):
        batch_x = self.x[index]
        batch_y = self.y[index]
        batch_z = self.z[index]
        return  batch_x, batch_y, batch_z
    
    def __len__(self):
        return self.x.size(0)

    
def gen_patches_nsa_dicom(file_r,file_i,NEX1r,NEX2r,NEX1i,NEX2i):
    patches_1 = []
    patches_2 = []
    patches_18 = []
    file_length = len(file_r)
    for i in range(file_length):
        img1r = torch.squeeze(torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(NEX1r[i]))))
        img2r = torch.squeeze(torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(NEX2r[i]))))
        imgr = torch.squeeze(torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(file_r[i]))))
        img1i = torch.squeeze(torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(NEX1i[i]))))
        img2i = torch.squeeze(torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(NEX2i[i]))))
        imgr = torch.squeeze(torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(file_i[i]))))
        l, h= img1r.shape
        for j in range(0, l-patch_size+1, stride):
            for k in range(0, h-patch_size+1, stride):
                xr = img1r[j:j+patch_size, k:k+patch_size] 
                yr = img2r[j:j+patch_size, k:k+patch_size]
                xi = img1i[j:j+patch_size, k:k+patch_size] 
                yi = img2i[j:j+patch_size, k:k+patch_size]
                x = imgr[j:j+patch_size, k:k+patch_size] 
                y = imgi[j:j+patch_size, k:k+patch_size] 
                patches_1.append(torch.stack((xr,xi),2))
                patches_2.append(torch.stack((yr,yi),2))
                patches_18.append(torch.stack((x,y),2))
    patches_1 = torch.stack(patches_1,0)
    patches_1 = patches_1.numpy()
    patches_2 = torch.stack(patches_2,0)
    patches_2 = patches_2.numpy()
    patches_18 = torch.stack(patches_18,0)
    patches_18 = patches_18.numpy()
                               
    return patches_1,patches_2,patches_18


def datagenerator(data_dir,folder): 
    file_r = []
    file_i = []
    NEX1r = []
    NEX2r = []
    NEX1i = []
    NEX2i = []
    fo = ['M','R','I','NEX1','NEX2']
    for file in folder:
        files_r = os.listdir(os.path.join(data_dir,file,fo[1]))
        files_i = os.listdir(os.path.join(data_dir,file,fo[2]))
        for path in files_r:
            full_R_path = os.path.join(data_dir,file,fo[1],path)
            NEX1r_path = os.path.join(data_dir,file,fo[3],fo[1],path)
            NEX2r_path = os.path.join(data_dir,file,fo[4],fo[1],path)
            file_r.append(full_R_path)
            NEX1r.append(NEX1r_path)
            NEX2r.append(NEX2r_path)
        for path in files_i:
            full_I_path = os.path.join(data_dir,file,fo[2],path)
            NEX1i_path = os.path.join(data_dir,file,fo[3],fo[2],path)
            NEX2i_path = os.path.join(data_dir,file,fo[4],fo[2],path)
            file_i.append(full_I_path)
            NEX1i.append(NEX1i_path)
            NEX2i.append(NEX2i_path)
    data_1,data_2,data_18 = gen_patches_nsa_dicom(file_r,file_i,NEX1r,NEX2r,NEX1i,NEX2i)
    data_1 = np.array(data_1, dtype='float32')
    data_2 = np.array(data_2, dtype='float32')
    data_18 = np.array(data_18, dtype='float32')
    print('^_^-Training data finished-^_^')

    return data_1,data_2,data_18


if __name__ == '__main__': 

    data = datagenerator()