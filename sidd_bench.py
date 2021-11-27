from numpy.lib.function_base import average
import torch
from torch.utils.data import dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchsummaryX import summary
import torchvision.datasets as datasets
import mat73
# from datetime import datetime
# from torchsummary import summary
import os
from math import log10
import numpy as np
import torch.nn as nn
import sys
import scipy.io
from PIL import Image
from collections import OrderedDict
from datasets import benchmark_dataset as test_loader
# from dncnn_model import DnCNN
# from dncnn_model_noisemap import DnCNN
# from dncnn_noisemap_plus2 import DnCNN
# from attention2 import DnCNN
# from dncnn_noisemap_plus_middleain import DnCNN

# from dataset import benchmark_dataset as test_loader


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def np_to_pil(np_imgs):
    img_num = np_imgs.shape[0]
    channel_num = np_imgs.shape[1]
    ar = np.clip(np_imgs*255, 0, 255).astype(np.uint8)
    pil_imgs = []
    for i in range(img_num):
        if channel_num == 1:
            img = ar[i][0]
        else:
            img = ar[i].transpose(1, 2, 0)
        pil_imgs.append(Image.fromarray(img))

    return pil_imgs


def denoise_patches(eval_loader, model, patches):
    imgs = [None] * 40
    patches_denoised = [None] * 32

    denoised_patches = torch.empty(len(eval_loader), len(patches), 3, 256, 256)
    for i, noise_img in enumerate(eval_loader):
        print(f"Image: [{i}]")
        noise_img = noise_img.cuda()
        for p, patch in enumerate(patches):
            noise_patch = noise_img[
                :,
                :,
                patch[0] : patch[0] + patch[2],
                patch[1] : patch[1] + patch[3]
            ]
            # denoised_patches[i, p, :, :] = model(noise_patch)
            # denoised_patches[i, p, :, :] = noise_patch[0]
            noise_patch = noise_patch * 255
            noise_patch = noise_patch.squeeze()
            noise_patch = torch.permute(noise_patch, (1, 2, 0))
            noise_patch = noise_patch.cpu().detach().numpy().astype('uint8')
            patches_denoised[p] = noise_patch
        imgs[i] = patches_denoised
    return imgs


if __name__ == '__main__':
    matlab_run = mat73.loadmat('Submit/SubmitSrgb.mat')
    # model_path = "/home/bledc/my_remote_folder/denoiser/models/Nov30_dncnnRGB_noisemap_aindSingle_03_19_53/"
    # model_path = model_path + model_epoch
    # eval_dataset = test_loader(
    #     '/home/bledc/dataset/SIDD/SIDD_Small_sRGB_Only/Data/test_set/vol_1', 1, rgb=True)

    # eval_loader = torch.utils.data.DataLoader(
    #     eval_dataset, batch_size=1,
    #     shuffle=True, num_workers=1,
    #     pin_memory=True, drop_last=True)
    crop_locations = scipy.io.loadmat('SIDD_Benchmark_Data/BenchmarkBlocks32.mat')
    crop_locations = crop_locations['BenchmarkBlocks32']
    # Convert matlab indexing to python
    mask = np.full((crop_locations.shape[0]) , 1)
    crop_locations[:, 0] = crop_locations[:, 0] - mask
    crop_locations[:, 1] = crop_locations[:, 1] - mask
    patches = mat73.loadmat('Submit/SubmitSrgb.mat')
    imgs = patches['DenoisedBlocksSrgb']

    model_epoch = "model_epoch_5000.pt"
    model_path = "/home/bledc/my_remote_folder/denoiser/models/Nov30_dncnnRGB_noisemap_aindSingle_03_19_53/"
    model_path = model_path + model_epoch

    # eval_dataset = test_loader(
    #     '/home/clement/Documents/light_code/sidd_eval/SIDD_Benchmark_Data', 1, rgb=True)
    eval_dataset = test_loader(
        '/home/clement/Documents/light_code/sidd_eval/SIDD_Benchmark_Data', 1, rgb=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1,
        shuffle=False, num_workers=1,
        pin_memory=True, drop_last=False)
    model = "placeholder"

    denoised_patches = denoise_patches(eval_loader, model, crop_locations)
    # Needs to be in nested list format like matlab version....

    sidd_dict = {
        "DenoisedBlocksSrgb": denoised_patches,
        "OptionalData": "",
        "TimeMPSrgb": ""
    }