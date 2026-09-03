import torch
import torch.nn as nn
import torch.nn.functional as F


class Pspnet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(Pspnet, self).__init__()

        # --------------------------
        # 骨干网络（示例使用ResNet50结构）
        # --------------------------
        def conv3x3(in_planes, out_planes, stride=1):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)

        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = conv3x3(planes, planes, stride)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(planes * self.expansion)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
                out = self.relu(out)
                return out

        # 简化的ResNet骨干
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建残差块
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=1)  # 保持高分辨率

        # --------------------------
        # 金字塔池化模块
        # --------------------------
        class PyramidPooling(nn.Module):
            def __init__(self, in_channels):
                super(PyramidPooling, self).__init__()
                self.pool_sizes = [1, 2, 3, 6]
                out_channels = in_channels // len(self.pool_sizes)

                self.convs = nn.ModuleList([
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(pool_size),
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    ) for pool_size in self.pool_sizes
                ])

                self.final_conv = nn.Sequential(
                    nn.Conv2d(in_channels + out_channels * len(self.pool_sizes), out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                input_size = x.size()[2:]
                features = [x]
                for conv in self.convs:
                    pooled = conv[0](x)
                    pooled = conv[1](pooled)
                    pooled = conv[2](pooled)
                    pooled = F.interpolate(pooled, size=input_size, mode='bilinear', align_corners=True)
                    features.append(pooled)
                x = torch.cat(features, dim=1)
                x = self.final_conv(x)
                return x

        # --------------------------
        # 输出层
        # --------------------------
        self.psp = PyramidPooling(2048)  # ResNet50最后一层通道数
        self.final_conv = nn.Conv2d(512, out_ch, kernel_size=1)  # 最终输出通道调整

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 记录原始尺寸
        input_size = x.size()[2:]

        # 骨干网络
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 输出尺寸为原图1/8

        # 金字塔池化
        x = self.psp(x)

        # 最终卷积
        x = self.final_conv(x)

        # 上采样到输入尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

        return x

