import pandas as pd
import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class SegDataset(Dataset):
    def __init__(self, workdir, path_csv, raw_transform = None, mask_transform = None):
        self.workdir = workdir
        self.data = pd.read_csv(os.path.join(self.workdir, path_csv))
        self.raw_transform = raw_transform
        self.mask_transform = mask_transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        raw = read_image(os.path.join(self.workdir, self.data.iloc[index, 0])) / 255
        mask = (read_image(os.path.join(self.workdir, self.data.iloc[index, 1])) / 255).long()       

        if self.raw_transform:
            raw = self.raw_transform(raw)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return raw, mask
        
class KaptchaDataset(Dataset):
    def __init__(self, workdir, images_type, data_csv = 'captcha_data.csv', image_transform = None):
        self.workdir = workdir
        self.images_type = images_type
        self.image_transform = image_transform
        data = pd.read_csv(os.path.join(self.workdir, data_csv), dtype = {'solution' : str})
        self.images_path_sol = data[data['image_path'].str.startswith(self.images_type)]
        self.images_path_sol['split_sol'] = self.images_path_sol['solution'].apply(lambda x: [int(smb) for smb in list(x)])
        
    def __len__(self):
        return len(self.images_path_sol)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = read_image(os.path.join(self.workdir, self.images_path_sol.iloc[idx,0])) / 255
        solution = torch.tensor(self.images_path_sol.iloc[idx, 2])

        if self.image_transform:
            img = self.image_transform(img)
            
        return img, solution