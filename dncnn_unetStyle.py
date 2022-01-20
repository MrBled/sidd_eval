from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import basicblocks as B


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fcn(x)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class down_block(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super().__init__()
        # activation = []
        for t in mode:
            if t == 'R':
                act = nn.ReLU(inplace=True)
            elif t == 'r':
                act = nn.ReLU(inplace=False)
            elif t == 'L':
                act = nn.LeakyReLU( inplace=True)
            elif t == 'l':
                act = nn.LeakyReLU(inplace=False)

        self.strided_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.strided_conv(x)


class up_block(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_bilinear_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, encoder_x):
        upsampled =  self.up(x)
        concat = torch.cat([upsampled, encoder_x], 1)
        out = self.up_bilinear_conv(concat)
        return out


class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super().__init__()
        # activation = []
        for t in mode:
            if t == 'R':
                act = nn.ReLU(inplace=True)
            elif t == 'r':
                act = nn.ReLU(inplace=False)
            elif t == 'L':
                act = nn.LeakyReLU( inplace=True)
            elif t == 'l':
                act = nn.LeakyReLU(inplace=False)

        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.maxpool_conv(x)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class DnCNN(nn.Module):
    def __init__(self, in_nc=6, out_nc=3, nc=64, nb=20, act_mode='BR'):
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode
        bias = False
        self.fcn = FCN()
        head1 = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        head2 = conv(nc, nc, mode='C'+act_mode, bias=bias)
        self.head = sequential(head1, head2)
        self.down1 = down_block(nc, nc, act_mode)
        self.down2 = down_block(nc, nc, act_mode)
        btlnck = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(8)]
        self.bottleneck = sequential(*btlnck)
        self.noisemap = conv(in_nc, nc, mode='C'+act_mode, bias=bias)
        self.up1 = up_block(nc*2, nc, mode=act_mode)
        self.up2 = up_block(nc*2, nc, mode=act_mode)
        self.tail = conv(nc, out_nc, mode='C', bias=bias)
        self.downsample = nn.MaxPool2d(2)


    def forward(self, x, train_mode=True):
        # noise_map = torch.full((x.size(0), 1, x.size(2), x.size(3)), stand_dev)
        # x_concat = torch.cat((x, noise_map), 1)
        noise_level = self.fcn(x)
        noise_level_down1 = self.downsample(noise_level)
        noise_level_down2 = self.downsample(noise_level_down1)
        noisemap_bottleneck = self.noisemap(noise_level_down2)

        in_layer = self.head(x)                 # x2 convs
        down1 = self.down1(in_layer)            # x2 convs
        down2 = self.down2(down1)               # x2 convs

        bottle_neck = self.bottleneck(down2 + noisemap_bottleneck) # add noisemap into net        # x8 convs
        up1 =  self.up1(bottle_neck, down1)     # x2 convs
        up2 = self.up2(up1, in_layer)           # x2 convs
        out = self.tail(up2)                    # x1 convs

        return noise_level, out


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        l2_loss = F.mse_loss(out_image, gt_image)

        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = l2_loss +  0.5 * asym_loss + 0.05 * tvloss

        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]