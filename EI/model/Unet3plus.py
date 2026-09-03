import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础卷积块 (VGG风格的双卷积)
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UNet3Plus(nn.Module):
    """
    UNet3+ 的实现:
    论文: https://arxiv.org/pdf/2004.08790.pdf
    该实现遵循 __init__(self, in_ch=3, out_ch=1) 接口。
    基础通道数在内部设定。
    """
    def __init__(self, in_ch=3, out_ch=1): # 严格遵守指定的接口
        super(UNet3Plus, self).__init__()

        # ---- 在内部设定基础通道数 ----
        filters_base = 64 # 基础通道数，等同于原 UNet++ 代码中的 n1
        # --------------------------

        # 根据基础通道数计算各层通道数
        filters = [filters_base, filters_base * 2, filters_base * 4, filters_base * 8, filters_base * 16]
        # [64, 128, 256, 512, 1024]

        # UNet3+ 相关参数
        # 每个连接路径在拼接前，通道数被统一为 cat_channels
        cat_channels = filters[0] # 这里设为与 filters_base 相同，即 64
        # 每个解码器层接收的拼接路径数量 (来自编码器的所有层 + 来自更深解码器的所有层)
        cat_blocks = 5 # 对应 5 个尺度的输入源
        # 解码器各层拼接后的总通道数 (进行最终3x3卷积前的通道数)
        up_conv_channels = cat_channels * cat_blocks # 64 * 5 = 320

        '''编码器 (Encoder)'''
        # 层 1
        self.conv_enc1 = VGGBlock(in_ch, filters[0], filters[0]) # 输入通道数为 in_ch
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 层 2
        self.conv_enc2 = VGGBlock(filters[0], filters[1], filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 层 3
        self.conv_enc3 = VGGBlock(filters[1], filters[2], filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 层 4
        self.conv_enc4 = VGGBlock(filters[2], filters[3], filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 层 5 (Bottleneck)
        self.conv_enc5 = VGGBlock(filters[3], filters[4], filters[4])

        '''解码器 (Decoder) - 全尺度连接'''

        # --- 解码器第 4 层 (对应编码器 h4 的尺度) ---
        # 处理来自 h1 (编码器第1层) 的连接 -> 下采样到 h4 尺度
        self.h1_pt_dec4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_conv_dec4 = nn.Conv2d(filters[0], cat_channels, 3, padding=1)
        self.h1_bn_dec4 = nn.BatchNorm2d(cat_channels)
        self.h1_relu_dec4 = nn.ReLU(inplace=True)
        # 处理来自 h2 (编码器第2层) 的连接 -> 下采样到 h4 尺度
        self.h2_pt_dec4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_conv_dec4 = nn.Conv2d(filters[1], cat_channels, 3, padding=1)
        self.h2_bn_dec4 = nn.BatchNorm2d(cat_channels)
        self.h2_relu_dec4 = nn.ReLU(inplace=True)
        # 处理来自 h3 (编码器第3层) 的连接 -> 下采样到 h4 尺度
        self.h3_pt_dec4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_conv_dec4 = nn.Conv2d(filters[2], cat_channels, 3, padding=1)
        self.h3_bn_dec4 = nn.BatchNorm2d(cat_channels)
        self.h3_relu_dec4 = nn.ReLU(inplace=True)
        # 处理来自 h4 (编码器第4层) 的连接 -> 尺度相同
        self.h4_conv_dec4 = nn.Conv2d(filters[3], cat_channels, 3, padding=1)
        self.h4_bn_dec4 = nn.BatchNorm2d(cat_channels)
        self.h4_relu_dec4 = nn.ReLU(inplace=True)
        # 处理来自 h5 (编码器第5层/Bottleneck) 的连接 -> 上采样到 h4 尺度
        self.h5_upsample_dec4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # h5->h4
        self.h5_conv_dec4 = nn.Conv2d(filters[4], cat_channels, 3, padding=1)
        self.h5_bn_dec4 = nn.BatchNorm2d(cat_channels)
        self.h5_relu_dec4 = nn.ReLU(inplace=True)
        # 拼接后进行卷积融合
        self.conv_dec4 = nn.Conv2d(up_conv_channels, up_conv_channels, 3, padding=1)
        self.bn_dec4 = nn.BatchNorm2d(up_conv_channels)
        self.relu_dec4 = nn.ReLU(inplace=True)

        # --- 解码器第 3 层 (对应编码器 h3 的尺度) ---
        # 处理来自 h1 -> 下采样到 h3 尺度
        self.h1_pt_dec3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_conv_dec3 = nn.Conv2d(filters[0], cat_channels, 3, padding=1)
        self.h1_bn_dec3 = nn.BatchNorm2d(cat_channels)
        self.h1_relu_dec3 = nn.ReLU(inplace=True)
        # 处理来自 h2 -> 下采样到 h3 尺度
        self.h2_pt_dec3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_conv_dec3 = nn.Conv2d(filters[1], cat_channels, 3, padding=1)
        self.h2_bn_dec3 = nn.BatchNorm2d(cat_channels)
        self.h2_relu_dec3 = nn.ReLU(inplace=True)
        # 处理来自 h3 -> 尺度相同
        self.h3_conv_dec3 = nn.Conv2d(filters[2], cat_channels, 3, padding=1)
        self.h3_bn_dec3 = nn.BatchNorm2d(cat_channels)
        self.h3_relu_dec3 = nn.ReLU(inplace=True)
        # 处理来自 dec4 (解码器第4层输出) -> 上采样到 h3 尺度
        self.h4_upsample_dec3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # dec4->h3
        self.h4_conv_dec3 = nn.Conv2d(up_conv_channels, cat_channels, 3, padding=1) # 输入来自 dec4 输出 (320通道)
        self.h4_bn_dec3 = nn.BatchNorm2d(cat_channels)
        self.h4_relu_dec3 = nn.ReLU(inplace=True)
        # 处理来自 h5 -> 上采样到 h3 尺度
        self.h5_upsample_dec3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # h5->h3
        self.h5_conv_dec3 = nn.Conv2d(filters[4], cat_channels, 3, padding=1)
        self.h5_bn_dec3 = nn.BatchNorm2d(cat_channels)
        self.h5_relu_dec3 = nn.ReLU(inplace=True)
        # 拼接后进行卷积融合
        self.conv_dec3 = nn.Conv2d(up_conv_channels, up_conv_channels, 3, padding=1)
        self.bn_dec3 = nn.BatchNorm2d(up_conv_channels)
        self.relu_dec3 = nn.ReLU(inplace=True)

        # --- 解码器第 2 层 (对应编码器 h2 的尺度) ---
        # 处理来自 h1 -> 下采样到 h2 尺度
        self.h1_pt_dec2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_conv_dec2 = nn.Conv2d(filters[0], cat_channels, 3, padding=1)
        self.h1_bn_dec2 = nn.BatchNorm2d(cat_channels)
        self.h1_relu_dec2 = nn.ReLU(inplace=True)
        # 处理来自 h2 -> 尺度相同
        self.h2_conv_dec2 = nn.Conv2d(filters[1], cat_channels, 3, padding=1)
        self.h2_bn_dec2 = nn.BatchNorm2d(cat_channels)
        self.h2_relu_dec2 = nn.ReLU(inplace=True)
        # 处理来自 dec3 -> 上采样到 h2 尺度
        self.h3_upsample_dec2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # dec3->h2
        self.h3_conv_dec2 = nn.Conv2d(up_conv_channels, cat_channels, 3, padding=1) # 输入来自 dec3 输出
        self.h3_bn_dec2 = nn.BatchNorm2d(cat_channels)
        self.h3_relu_dec2 = nn.ReLU(inplace=True)
        # 处理来自 dec4 -> 上采样到 h2 尺度
        self.h4_upsample_dec2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # dec4->h2
        self.h4_conv_dec2 = nn.Conv2d(up_conv_channels, cat_channels, 3, padding=1) # 输入来自 dec4 输出
        self.h4_bn_dec2 = nn.BatchNorm2d(cat_channels)
        self.h4_relu_dec2 = nn.ReLU(inplace=True)
        # 处理来自 h5 -> 上采样到 h2 尺度
        self.h5_upsample_dec2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) # h5->h2
        self.h5_conv_dec2 = nn.Conv2d(filters[4], cat_channels, 3, padding=1)
        self.h5_bn_dec2 = nn.BatchNorm2d(cat_channels)
        self.h5_relu_dec2 = nn.ReLU(inplace=True)
        # 拼接后进行卷积融合
        self.conv_dec2 = nn.Conv2d(up_conv_channels, up_conv_channels, 3, padding=1)
        self.bn_dec2 = nn.BatchNorm2d(up_conv_channels)
        self.relu_dec2 = nn.ReLU(inplace=True)

        # --- 解码器第 1 层 (对应编码器 h1 的尺度 - 输出层) ---
        # 处理来自 h1 -> 尺度相同
        self.h1_conv_dec1 = nn.Conv2d(filters[0], cat_channels, 3, padding=1)
        self.h1_bn_dec1 = nn.BatchNorm2d(cat_channels)
        self.h1_relu_dec1 = nn.ReLU(inplace=True)
        # 处理来自 dec2 -> 上采样到 h1 尺度
        self.h2_upsample_dec1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # dec2->h1
        self.h2_conv_dec1 = nn.Conv2d(up_conv_channels, cat_channels, 3, padding=1) # 输入来自 dec2 输出
        self.h2_bn_dec1 = nn.BatchNorm2d(cat_channels)
        self.h2_relu_dec1 = nn.ReLU(inplace=True)
        # 处理来自 dec3 -> 上采样到 h1 尺度
        self.h3_upsample_dec1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # dec3->h1
        self.h3_conv_dec1 = nn.Conv2d(up_conv_channels, cat_channels, 3, padding=1) # 输入来自 dec3 输出
        self.h3_bn_dec1 = nn.BatchNorm2d(cat_channels)
        self.h3_relu_dec1 = nn.ReLU(inplace=True)
        # 处理来自 dec4 -> 上采样到 h1 尺度
        self.h4_upsample_dec1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) # dec4->h1
        self.h4_conv_dec1 = nn.Conv2d(up_conv_channels, cat_channels, 3, padding=1) # 输入来自 dec4 输出
        self.h4_bn_dec1 = nn.BatchNorm2d(cat_channels)
        self.h4_relu_dec1 = nn.ReLU(inplace=True)
        # 处理来自 h5 -> 上采样到 h1 尺度
        self.h5_upsample_dec1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True) # h5->h1
        self.h5_conv_dec1 = nn.Conv2d(filters[4], cat_channels, 3, padding=1)
        self.h5_bn_dec1 = nn.BatchNorm2d(cat_channels)
        self.h5_relu_dec1 = nn.ReLU(inplace=True)
        # 拼接后进行卷积融合
        self.conv_dec1 = nn.Conv2d(up_conv_channels, up_conv_channels, 3, padding=1)
        self.bn_dec1 = nn.BatchNorm2d(up_conv_channels)
        self.relu_dec1 = nn.ReLU(inplace=True)

        '''最终输出层'''
        # 1x1 卷积将最终特征图通道数调整为 out_ch
        self.final_conv = nn.Conv2d(up_conv_channels, out_ch, kernel_size=1)


    def forward(self, x):
        '''编码器路径'''
        h1 = self.conv_enc1(x)     # h1 shape: [B, 64, H, W]
        h2 = self.pool1(h1)
        h2 = self.conv_enc2(h2)    # h2 shape: [B, 128, H/2, W/2]
        h3 = self.pool2(h2)
        h3 = self.conv_enc3(h3)    # h3 shape: [B, 256, H/4, W/4]
        h4 = self.pool3(h3)
        h4 = self.conv_enc4(h4)    # h4 shape: [B, 512, H/8, W/8]
        h5 = self.pool4(h4)
        h5 = self.conv_enc5(h5)    # h5 shape: [B, 1024, H/16, W/16] (Bottleneck)

        '''解码器路径 + 全尺度特征融合'''
        # --- 解码器第 4 层 ---
        # 对来自不同尺度的输入进行处理和通道标准化
        h1_dec4 = self.h1_relu_dec4(self.h1_bn_dec4(self.h1_conv_dec4(self.h1_pt_dec4(h1))))
        h2_dec4 = self.h2_relu_dec4(self.h2_bn_dec4(self.h2_conv_dec4(self.h2_pt_dec4(h2))))
        h3_dec4 = self.h3_relu_dec4(self.h3_bn_dec4(self.h3_conv_dec4(self.h3_pt_dec4(h3))))
        h4_dec4 = self.h4_relu_dec4(self.h4_bn_dec4(self.h4_conv_dec4(h4)))
        h5_dec4 = self.h5_relu_dec4(self.h5_bn_dec4(self.h5_conv_dec4(self.h5_upsample_dec4(h5))))
        # 拼接所有标准化后的特征图
        dec4 = torch.cat((h1_dec4, h2_dec4, h3_dec4, h4_dec4, h5_dec4), 1) # shape: [B, 320, H/8, W/8]
        # 进行卷积融合
        dec4 = self.relu_dec4(self.bn_dec4(self.conv_dec4(dec4))) # shape: [B, 320, H/8, W/8]

        # --- 解码器第 3 层 ---
        h1_dec3 = self.h1_relu_dec3(self.h1_bn_dec3(self.h1_conv_dec3(self.h1_pt_dec3(h1))))
        h2_dec3 = self.h2_relu_dec3(self.h2_bn_dec3(self.h2_conv_dec3(self.h2_pt_dec3(h2))))
        h3_dec3 = self.h3_relu_dec3(self.h3_bn_dec3(self.h3_conv_dec3(h3)))
        h4_dec3 = self.h4_relu_dec3(self.h4_bn_dec3(self.h4_conv_dec3(self.h4_upsample_dec3(dec4)))) # 输入来自 dec4
        h5_dec3 = self.h5_relu_dec3(self.h5_bn_dec3(self.h5_conv_dec3(self.h5_upsample_dec3(h5))))
        dec3 = torch.cat((h1_dec3, h2_dec3, h3_dec3, h4_dec3, h5_dec3), 1) # shape: [B, 320, H/4, W/4]
        dec3 = self.relu_dec3(self.bn_dec3(self.conv_dec3(dec3))) # shape: [B, 320, H/4, W/4]

        # --- 解码器第 2 层 ---
        h1_dec2 = self.h1_relu_dec2(self.h1_bn_dec2(self.h1_conv_dec2(self.h1_pt_dec2(h1))))
        h2_dec2 = self.h2_relu_dec2(self.h2_bn_dec2(self.h2_conv_dec2(h2)))
        h3_dec2 = self.h3_relu_dec2(self.h3_bn_dec2(self.h3_conv_dec2(self.h3_upsample_dec2(dec3)))) # 输入来自 dec3
        h4_dec2 = self.h4_relu_dec2(self.h4_bn_dec2(self.h4_conv_dec2(self.h4_upsample_dec2(dec4)))) # 输入来自 dec4
        h5_dec2 = self.h5_relu_dec2(self.h5_bn_dec2(self.h5_conv_dec2(self.h5_upsample_dec2(h5))))
        dec2 = torch.cat((h1_dec2, h2_dec2, h3_dec2, h4_dec2, h5_dec2), 1) # shape: [B, 320, H/2, W/2]
        dec2 = self.relu_dec2(self.bn_dec2(self.conv_dec2(dec2))) # shape: [B, 320, H/2, W/2]

        # --- 解码器第 1 层 ---
        h1_dec1 = self.h1_relu_dec1(self.h1_bn_dec1(self.h1_conv_dec1(h1)))
        h2_dec1 = self.h2_relu_dec1(self.h2_bn_dec1(self.h2_conv_dec1(self.h2_upsample_dec1(dec2)))) # 输入来自 dec2
        h3_dec1 = self.h3_relu_dec1(self.h3_bn_dec1(self.h3_conv_dec1(self.h3_upsample_dec1(dec3)))) # 输入来自 dec3
        h4_dec1 = self.h4_relu_dec1(self.h4_bn_dec1(self.h4_conv_dec1(self.h4_upsample_dec1(dec4)))) # 输入来自 dec4
        h5_dec1 = self.h5_relu_dec1(self.h5_bn_dec1(self.h5_conv_dec1(self.h5_upsample_dec1(h5))))
        dec1 = torch.cat((h1_dec1, h2_dec1, h3_dec1, h4_dec1, h5_dec1), 1) # shape: [B, 320, H, W]
        dec1 = self.relu_dec1(self.bn_dec1(self.conv_dec1(dec1))) # shape: [B, 320, H, W]

        '''最终输出'''
        output = self.final_conv(dec1) # shape: [B, out_ch, H, W]

        return output
