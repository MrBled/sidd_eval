from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.strided_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.skip_cn = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, mode='CBR')

    def forward(self, x):
        conv_block = self.strided_conv(x)
        if conv_block.size() == x.size():
            out = x + conv_block
            return out
        else:
            return conv_block

class dncnn_block(nn.Module):
    def __init__(self, in_channels, nc, out_channels):
        super().__init__()
        self.resblock1 = resblock(in_channels, nc)
        self.resblock2 = resblock(nc, nc)
        self.resblock3 = resblock(nc, nc)
        self.resblock4 = resblock(nc, nc)
        self.resblock5 = resblock(nc, nc)
        self.resblock6 = resblock(nc, nc)
        self.resblock7 = resblock(nc, nc)
        self.resblock8 = resblock(nc, nc)
        self.resblock9 = resblock(nc, out_channels)


    def forward(self, x):
        layer1 = self.resblock1(x)
        layer2 = self.resblock2(layer1)
        layer3 = self.resblock3(layer2)
        layer4 = self.resblock4(layer3)
        layer5 = self.resblock5(layer4)
        layer6 = self.resblock6(layer5)
        layer7 = self.resblock7(layer6)
        layer8 = self.resblock8(layer7)
        layer9 = self.resblock9(layer8)

        return layer9


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
        # self.dncnn1 = dncnn_block(nc, nc, out_nc)
        # self.dncnn2 = dncnn_block(nc, nc, out_nc)
        # self.head1 = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        # self.tail1 = conv(nc, out_nc, mode='C', bias=bias)
        # self.head2 = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        # self.tail2 = conv(nc, out_nc, mode='C', bias=bias)

        dncnn1 = dncnn_block(nc, nc, nc)
        dncnn2 = dncnn_block(nc, nc, nc)
        head1 = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        tail1 = conv(nc, out_nc, mode='C', bias=bias)
        head2 = conv(9, nc, mode='C'+act_mode[-1], bias=bias)
        tail2 = conv(nc, out_nc, mode='C', bias=bias)

        self.model1 = sequential(head1, dncnn1, tail1)
        self.model2 = sequential(head2, dncnn2, tail2)

    def forward(self, x, train_mode=True):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], 1)

        level1_out = self.model1(concat_img)
        stage2_in = torch.cat([level1_out, x, noise_level], 1)
        level2_out = self.model2(stage2_in) + x

        # level1_out = self.dncnn1(concat_img)
        # stage2_in = torch.cat([level1_out, x, noise_level], 1)
        # level2_out = self.dncnn2(stage2_in) + x

        return noise_level, level1_out, level2_out


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, stage1_img, stage2_img,  gt_image, est_noise, gt_noise, if_asym):
        l2_loss_stage1 = F.mse_loss(stage1_img, gt_image)
        l2_loss_stage2 = F.mse_loss(stage2_img, gt_image)

        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = l2_loss_stage2 + (0.5 * l2_loss_stage1) + (0.5 * asym_loss) + (0.05 * tvloss)

        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]