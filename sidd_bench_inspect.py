from numpy.lib.function_base import average
import torch
from torch.utils.data import dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchsummaryX import summary
from PIL import Image
import torchvision.datasets as datasets
import mat73
# from datetime import datetime
# from torchsummary import summary
import os
from math import log10
import copy
import numpy as np
import torch.nn as nn
import sys
import scipy.io
from PIL import Image
from collections import OrderedDict
from datasets import benchmark_dataset as test_loader
# from dncnn_stacked import DnCNN
from noisemap_end import DnCNN
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

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

    for i, (noise_img, fname) in enumerate(eval_loader):
        print(f"Image: [{i}]")
        print(fname)
        noise_img = noise_img.cuda()
        for p, patch in enumerate(patches):
            noise_patch = noise_img[
                :,
                :,
                patch[0] : patch[0] + patch[2], #start to stop - 1
                patch[1] : patch[1] + patch[3]
            ]
            _, noise_patch = model(noise_patch)
            # noise_patch, _, _  = model(noise_patch)
            noise_patch = noise_patch.squeeze()
            # save_image(noise_patch, f'/home/bledc/dataset/checkout_ims/im_{i}_{p}.png')
            noise_patch = noise_patch * 255
            noise_patch = torch.clamp(noise_patch, 0, 255)
            noise_patch = torch.permute(noise_patch, (1, 2, 0))
            t_max = torch.max(noise_patch)
            t_min = torch.min(noise_patch)
            noise_patch = noise_patch.cpu().detach().numpy().astype('uint8')
            np_max = np.max(noise_patch)
            np_min = np.min(noise_patch)
            # print(f"torch min/max: {t_min:.2f}/{t_max:.2f} \t np min/max: {np_min}/{np_max}")
            noise_patch_im = Image.fromarray(noise_patch)
            noise_patch_im.save(f'/home/bledc/dataset/checkout_ims/im_{i}_{p}.png')
            patches_denoised[p] = noise_patch
            inner_list = copy.copy(patches_denoised)
        imgs[i] = inner_list
        # imgs[i] = patches_denoised
    return imgs


if __name__ == '__main__':
    # matlab_run = mat73.loadmat('Submit/SubmitSrgb.mat')

    model_epoch = "model_epoch_3060.pt"
    model_path = "/home/bledc/my_remote_folder/denoiser/models/Dec5_dncnn_noisemap_endMoreData_13_27_14/"
    model_path = model_path + model_epoch
    model = DnCNN(in_nc=3, out_nc=6, nc=64, nb=23, act_mode='BR')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(checkpoint['model_state_dict'])


    crop_locations = scipy.io.loadmat('SIDD_Benchmark_Code_v1.2/BenchmarkBlocks32.mat')
    crop_locations = crop_locations['BenchmarkBlocks32']
    # Convert matlab indexing to python
    mask = np.full((crop_locations.shape[0]) , 1)
    crop_locations[:, 0] = crop_locations[:, 0] - mask
    crop_locations[:, 1] = crop_locations[:, 1] - mask
    eval_dataset = test_loader(
        '/home/bledc/dataset/sidd_bench/SIDD_Benchmark_Data', 1, rgb=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1,
        shuffle=False, num_workers=1,
        pin_memory=True, drop_last=False)

    denoised_patches = denoise_patches(eval_loader, model, crop_locations)
    # Needs to be in nested list format like matlab version....

    sidd_dict = {
        "DenoisedBlocksSrgb": denoised_patches,
        "OptionalData": "",
        "TimeMPSrgb": ""
    }

    scipy.io.savemat("SubmitSrgb.mat", sidd_dict)