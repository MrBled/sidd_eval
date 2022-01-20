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


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, mode='C', bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, mode='C', bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


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


class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size, mode='C'))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, decoder_outs):
        x = self.orb1(x)
        x = x  + self.conv_dec1(decoder_outs)

        return x


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


def param_free_norm(x, epsilon=1e-5):
    x_mean = torch.mean(x)
    x_std = torch.sqrt(torch.var(x) + epsilon)
    return (x - x_mean) / x_std



class ain(nn.Module):
    def __init__(self, in_shape=3, out_shape=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_shape, out_shape, 5, padding=2, padding_mode='zeros')
        self.conv2 = nn.Conv2d(out_shape, out_shape, 3, padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(out_shape, out_shape, 3, padding=1, padding_mode='zeros')
        self.relu1 = nn.ReLU()

    def forward(self, noise_map, x_init, channels) :
        x = param_free_norm(x_init)
        tmp = self.conv1(noise_map)
        tmp = self.relu1(tmp)
        noise_map_gamma = self.conv2(tmp)
        noise_map_beta = self.conv3(tmp)
        x = x * (1 + noise_map_gamma) + noise_map_beta
        return x


class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, mode='C', bias=bias, padding=0)
        self.conv2 = conv(n_feat, 3, kernel_size, mode='C', bias=bias, padding=0)
        self.conv3 = conv(3, n_feat, kernel_size, mode='C', bias=bias, padding=0)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


class DnCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=20, act_mode='BR'):
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode
        reduction = 4
        kernel_size = 3
        scale_orsnetfeats = 32
        scale_unetfeats=48
        num_cab=8
        bias = False
        act=nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_nc, nc, mode='C', bias=bias), CAB(64, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2 = nn.Sequential(conv(in_nc, nc, mode='C', bias=bias), CAB(64, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3 = nn.Sequential(conv(in_nc, nc, mode='C', bias=bias), CAB(64, kernel_size, reduction, bias=bias, act=act))
        self.sam1 = SAM(nc, kernel_size=1, bias=bias)
        self.sam2 = SAM(nc, kernel_size=1, bias=bias)
        self.concat12 = conv(nc*2, nc, mode='C', bias=bias)
        self.concat23 = conv(nc*2, nc, mode='C', bias=bias)
        m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)
        # self.tail = conv(96, out_nc, mode='C', bias=bias)
        self.middle1 = sequential(*m_body)
        self.middle2 = sequential(*m_body)
        self.middle3 = sequential(*m_body, m_tail)
        # self.orsnet = ORSNet(nc, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)


    def forward(self, x, train_mode=True):
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        shallow_1 = self.shallow_feat1(x)
        result_1 = self.middle1(shallow_1)
        x1_sam_features, stage1_out = self.sam1(result_1, x)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        shallow_2 = self.shallow_feat2(x)
        x2_cat = self.concat12(torch.cat([shallow_2, x1_sam_features], 1))
        result_2 = self.middle2(x2_cat)
        x2_sam_features, stage2_out = self.sam2(result_2, x)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        shallow_3 = self.shallow_feat3(x)
        x3_cat = self.concat23(torch.cat([shallow_3, x2_sam_features], 1))
        result_3 = self.middle3(x3_cat)
        # final_3 = self.tail(result_3)
        final_result = result_3 + x
        return [final_result, stage2_out, stage1_out]


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