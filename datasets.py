
import os
import random
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import glob
from torch.utils.data import Dataset
from skimage import color, io
from utils import read_img, hwc_to_chw
from utils import read_img2

class benchmark_dataset(Dataset):
    def __init__(self, root_dir, sample_num, patch_size=128, rgb=False):
        self.patch_size = patch_size
        folders = glob.glob(root_dir + '/0*')
        folders.sort()
        # GET ALL FOLDERS IN DIR
        self.clean_fns = [None] * len(folders)
        for i in range(len(folders)):
            self.clean_fns[i] = []
        if not rgb:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(1),
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                ])
        # GET ALL FILES IN ALL FOLDERS ITERATIVELY
        for ind, folder in enumerate(folders):
            clean_imgs = glob.glob(folder + '/*NOISY_SRGB*')
            clean_imgs.sort()
            for clean_img in clean_imgs:
                self.clean_fns[ind % len(folders)].append(clean_img)

    def __len__(self):
        l = len(self.clean_fns)
        return l

    def __getitem__(self, idx):
        noisy_fn = self.clean_fns[idx][0]
        noisy_img = read_img(noisy_fn)
        noisy_img = self.transforms(noisy_img)
        return noisy_img, noisy_fn