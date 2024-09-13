import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random


class ModelNet40(Dataset):

    def __init__(self, args, mode='train', class_imbalanced=False):
        self.args = args
        self.mode = mode

        self.data_root = './data/'
        
        self.train_txt = os.path.join(self.data_root, args.dataset + '/train.txt')
        self.test_txt = os.path.join(self.data_root, args.dataset + '/test.txt')

        if mode == 'train':
            f = open(self.train_txt,'r')
            self.image = f.readlines() 
        else:
            f = open(self.test_txt,'r')
            self.image = f.readlines() 


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        img1_pth = self.image[idx].strip().split('\t')[0]
        img2_pth = self.image[idx].strip().split('\t')[0].replace('v001','v007')
        label = int(self.image[idx].strip().split('\t')[1])
        
        img1 = Image.open(img1_pth).convert('RGB')
        img2 = Image.open(img2_pth).convert('RGB')
        
        img1 = transform(img1).unsqueeze(1)
        img2 = transform(img2).unsqueeze(1)
        
#         images = torch.zeros((self.args.num_frame, 3, 224, 224))
#         for i in range(self.args.num_frame):
#             #  for i in select_index:
#             img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
#             img = transform(img)
#             images[i] = img

#         images = torch.permute(images, (1,0,2,3))

#         # label
#         label = self.label[idx]

        return img1, img2, label