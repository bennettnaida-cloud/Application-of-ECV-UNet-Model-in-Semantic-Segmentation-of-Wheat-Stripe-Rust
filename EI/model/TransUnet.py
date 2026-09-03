import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block as ViTBlock

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, D, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]
        x = self.norm(x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8, depth=4, patch_size=7):
        super().__init__()
        self.patch_size = patch_size
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size)**2, embed_dim))

    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        pos_embed = F.interpolate(
            self.pos_embed.reshape(1, (224//self.patch_size), (224//self.patch_size), -1).permute(0,3,1,2),
            size=(H, W),
            mode='bicubic'
        ).permute(0,2,3,1).reshape(1, -1, D)
        x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x

class TransUnet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, embed_dim=768, patch_size=6):  # 修改默认patch_size为7
        super().__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.Maxpool = nn.MaxPool2d(2, 2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.patch_embed = PatchEmbed(filters[3], embed_dim, patch_size)
        self.ViT = ViTEncoder(embed_dim, patch_size=patch_size)
        self.vit_proj = nn.Conv2d(embed_dim, filters[3], 1)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3] + filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2] + filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1] + filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, 1)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(filters[0], filters[0] // 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.Conv = nn.Conv2d(filters[0] // 2, out_ch, 1)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        vit_input = self.Maxpool(e4)
        vit_tokens = self.patch_embed(vit_input)
        vit_out = self.ViT(vit_tokens)

        B, N, D = vit_out.shape
        H, W = vit_input.shape[2] // self.patch_embed.patch_size, vit_input.shape[3] // self.patch_embed.patch_size
        vit_feature = vit_out.transpose(1, 2).view(B, D, H, W)
        vit_feature = self.vit_proj(vit_feature)

        d4 = self.Up4(vit_feature)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.Up_conv2(d2)

        # 新增上采样步骤
        out = self.final_upsample(d2)
        out = self.Conv(out)
        return out