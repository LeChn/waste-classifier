import numpy as np
from PIL import Image
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import os
import pandas as pd
import collections


class Waste_Data(Dataset):
    def __init__(self, name_list, train_jpg_dir, transform):
        super(Waste_Data, self).__init__()
        self.name_list = name_list
        self.transform = transform
        self.train_jpg_dir = train_jpg_dir

    def __getitem__(self, idx):
        filename = self.name_list[idx]
        image = Image.open(filename)
        imgList = [self.transform(image) for _ in range(2)]
        img = torch.cat(imgList, dim=0)
        return img

    def __len__(self):
        return len(self.name_list)


class Waste_Data_Finetune(Dataset):
    def __init__(self, name_list, label_csv, train_jpg_dir, transform):
        super(Waste_Data_Finetune, self).__init__()
        self.name_list = name_list
        self.transform = transform
        self.train_jpg_dir = train_jpg_dir
        self.label_csv = pd.read_csv(label_csv)
        self.label_csv.set_index(['Image'], inplace=True)
        self.classList = list(collections.Counter(self.label_csv.label).keys())

    def __getitem__(self, idx):
        filename = self.name_list[idx]
        filepath = os.path.join(self.train_jpg_dir, filename)
        label = torch.tensor(self.label_csv.loc[self.name_list[idx], self.classList])
        image = Image.open(filepath)
        img = self.transform(image)
        return img, label

    def __len__(self):
        return len(self.name_list)
