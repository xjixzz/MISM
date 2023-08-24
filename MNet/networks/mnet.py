import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from layers import Conv2d, ConvBlock, Conv3x3

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FPN4(nn.Module):
    """
    FPN aligncorners downsample 4x"""
    def __init__(self, base_channels, scale=0):
        super(FPN4, self).__init__()
        self.base_channels = base_channels
        self.scale = scale

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
        )
        
        final_chs = base_channels * 8

        if self.scale < 3:
            self.inner1 = nn.Conv2d(base_channels * 4, final_chs, 1, bias=True)
        if self.scale < 2:
            self.inner2 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        if self.scale < 1:
            self.inner3 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)
    
        if self.scale == 3:
            self.out = nn.Conv2d(final_chs, base_channels * 8, 1, bias=False)
        elif self.scale == 2:
            self.out = nn.Conv2d(final_chs, base_channels * 4, 3, padding=1, bias=False)
        elif self.scale == 1:
            self.out = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        else:
            self.out = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)


    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        intra_feat = conv3
        if self.scale < 3:
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv2)
        if self.scale < 2:
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv1)
        if self.scale < 1:
            intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(conv0)

        out = self.out(intra_feat)
        if self.scale == 3:
            return out, conv3
        elif self.scale == 2:
            return out, conv2
        elif self.scale == 1:
            return out, conv1
        else:
            return out, conv0
        
        
class reg3d(nn.Module):
    def __init__(self, in_channels, base_channels, down_size=3):
        super(reg3d, self).__init__()
        self.down_size = down_size
        self.conv0 = ConvBnReLU3D(in_channels, base_channels, kernel_size=3, pad=1)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels*2, kernel_size=3, stride=2, pad=1)
        self.conv2 = ConvBnReLU3D(base_channels*2, base_channels*2)
        if down_size >= 2:
            self.conv3 = ConvBnReLU3D(base_channels*2, base_channels*4, kernel_size=3, stride=2, pad=1)
            self.conv4 = ConvBnReLU3D(base_channels*4, base_channels*4)
        if down_size >= 3:
            self.conv5 = ConvBnReLU3D(base_channels*4, base_channels*8, kernel_size=3, stride=2, pad=1)
            self.conv6 = ConvBnReLU3D(base_channels*8, base_channels*8)
            self.conv7 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*4),
                nn.ReLU(inplace=True))
        if down_size >= 2:
            self.conv9 = nn.Sequential(
                nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(base_channels*2),
                nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True))
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, inputs):
        x = inputs.permute(0, 2, 1, 3, 4)  # B,D,C,H,W --> B,C,D,H,W
        if self.down_size==3:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
        elif self.down_size==2:
            conv0 = self.conv0(x)
            conv2 = self.conv2(self.conv1(conv0))
            x = self.conv4(self.conv3(conv2))
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
        else:
            conv0 = self.conv0(x)
            x = self.conv2(self.conv1(conv0))
            x = conv0 + self.conv11(x)

        x = self.prob(x)
        x = x.squeeze(1)  # B D H W

        return x  # B D H W
    

class convex_upsample_layer(nn.Module):
    def __init__(self, feature_dim, scale=2):
        super(convex_upsample_layer, self).__init__()
        self.scale = scale
        self.upsample_mask = nn.Sequential(
            nn.Conv2d(feature_dim, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, (2**scale)**2*9, 1, stride=1, padding=0, dilation=1, bias=False)
        )

    def forward(self, depth, feat):
        mask = self.upsample_mask(feat)
        return convex_upsample(depth, mask, self.scale)  # B H2 W2

def convex_upsample(depth, mask, scale=2):
    if len(depth.shape) == 3:
        B, H, W = depth.shape
        depth = depth.unsqueeze(1)
    else:
        B, _, H, W = depth.shape
    mask = mask.view(B, 9, 2**scale, 2**scale, H, W)
    mask = torch.softmax(mask, dim=1)

    up_ = F.unfold(depth, [3,3], padding=1)
    up_ = up_.view(B, 9, 1, 1, H, W)

    up_ = torch.sum(mask * up_, dim=1)  # B, 2**scale, 2**scale, H, W
    up_ = up_.permute(0, 3, 1, 4, 2)  # B H 2**scale W 2**scale
    return up_.reshape(B, 2**scale*H, 2**scale*W)

