# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


class FCN32s(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        # VGG16 backbone（集成编码器）
        self.vgg = models.vgg16(pretrained=False)
        # 适配输入通道[6](@ref)
        self.vgg.features[0] = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)

        # 解码器部分
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.classifier = nn.Conv2d(32, out_ch, kernel_size=1)

        # 统一激活函数和BN层[7](@ref)
        self.relu = nn.ReLU(inplace=True)
        self._init_bn_layers(512, 256, 128, 64, 32)

    def _init_bn_layers(self, *channels):
        for i, ch in enumerate(channels, 1):
            setattr(self, f'bn{i}', nn.BatchNorm2d(ch))

    def forward(self, x):
        # 编码器特征提取[2](@ref)
        x = self.vgg.features(x)

        # 解码过程[1](@ref)
        x = self.bn1(self.relu(self.deconv1(x)))  # 1/16
        x = self.bn2(self.relu(self.deconv2(x)))  # 1/8
        x = self.bn3(self.relu(self.deconv3(x)))  # 1/4
        x = self.bn4(self.relu(self.deconv4(x)))  # 1/2
        x = self.bn5(self.relu(self.deconv5(x)))  # original size
        return self.classifier(x)


class FCN16s(FCN32s):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__(in_ch, out_ch)
        # 增加pool4特征适配层[6](@ref)
        self.conv_pool4 = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, x):
        # 获取中间层特征[7](@ref)
        pool4 = self.vgg.features[:24](x)  # 1/16
        pool5 = self.vgg.features[24:](pool4)  # 1/32

        # 特征融合解码[2](@ref)
        x = self.relu(self.deconv1(pool5))  # 1/16
        x = self.bn1(x + self.conv_pool4(pool4))  # 特征相加
        x = self.bn2(self.relu(self.deconv2(x)))  # 1/8
        x = self.bn3(self.relu(self.deconv3(x)))  # 1/4
        x = self.bn4(self.relu(self.deconv4(x)))  # 1/2
        x = self.bn5(self.relu(self.deconv5(x)))  # original
        return self.classifier(x)


class FCN8s(FCN16s):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__(in_ch, out_ch)
        # 增加pool3特征适配层[6](@ref)
        self.conv_pool3 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x):
        # 多尺度特征提取[7](@ref)
        pool3 = self.vgg.features[:17](x)  # 1/8
        pool4 = self.vgg.features[17:24](pool3)  # 1/16
        pool5 = self.vgg.features[24:](pool4)  # 1/32

        # 分层特征融合[2](@ref)
        x = self.relu(self.deconv1(pool5))  # 1/16
        x = self.bn1(x + self.conv_pool4(pool4))  # pool4融合
        x = self.relu(self.deconv2(x))  # 1/8
        x = self.bn2(x + self.conv_pool3(pool3))  # pool3融合
        x = self.bn3(self.relu(self.deconv3(x)))  # 1/4
        x = self.bn4(self.relu(self.deconv4(x)))  # 1/2
        x = self.bn5(self.relu(self.deconv5(x)))  # original
        return self.classifier(x)
