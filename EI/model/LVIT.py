from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from timm.models.vision_transformer import Block as ViTBlock
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from timm.layers import use_fused_attn

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class LinearAttention(nn.Module):
    fused_attn: Final[bool]  # 虽然当前未使用，但保持结构统一

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,  # 保留参数但不使用
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,  # 保持参数结构
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim  ** -0.5
        self.fused_attn = False  # 线性注意力不支持融合实现

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, N, D]

        # 保持线性注意力核心计算
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        x = torch.einsum('b h n d, b h d e -> b h n e', q, context)

        # 保持输出结构一致
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LViTBlock(ViTBlock):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        # 替换原生的多头注意力为线性注意力
        self.attn = LinearAttention(embed_dim, num_heads)


class ViTEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads=8, depth=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))

        # 使用自定义的 CustomViTBlock
        self.blocks = nn.ModuleList([
            LViTBlock(embed_dim, num_heads) for _ in range(depth)  # 替换此处
        ])

    def forward(self, x):
        # 原forward代码保持不变
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        pos_embed = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear')
        x = x + pos_embed
        x = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class LVIT(nn.Module):
    """集成ViT的改进版U-Net"""

    def __init__(self, in_ch=3, out_ch=1):
        super(LVIT, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 编码器部分（前四层保持CNN）
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        # 用ViT替换原第五层卷积 [1](@ref)
        self.ViT_Encoder = ViTEncoder(
            in_channels=filters[3],
            embed_dim=filters[4],
            num_heads=8,
            depth=4
        )

        # 解码器部分（保持U-Net结构）
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        # 编码阶段
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        # ViT编码层 [1](@ref)
        e5 = self.Maxpool(e4)
        e5 = self.ViT_Encoder(e5)  # 输出形状保持[bs,1024,8,8]

        # 解码阶段
        d5 = self.Up5(e5)
        d5 = F.interpolate(d5, size=e4.shape[2:], mode='bilinear')
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear')
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear')
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear')
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out
