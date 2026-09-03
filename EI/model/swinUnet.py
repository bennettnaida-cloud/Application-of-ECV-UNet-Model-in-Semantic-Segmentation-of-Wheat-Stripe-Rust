import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange # Often used for reshaping in transformers
from typing import Optional, List, Tuple
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_ # Example import if using timm

class Mlp(nn.Module):
    # Placeholder - Replace with actual implementation
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        # Simplified forward
        return self.drop(self.fc2(self.act(self.fc1(x))))

def window_partition(x, window_size: int):
    """ Placeholder Helper Function """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """ Placeholder Helper Function """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
     # Placeholder - Replace with actual implementation (including relative pos bias)
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size) # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Relative position bias setup omitted for placeholder

    def forward(self, x, mask=None):
        # Simplified forward - Needs real implementation
        B_, N, C = x.shape # N = window_size * window_size
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Apply mask if provided (for shifted windows)
        # Apply relative position bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Placeholder for DropPath (needed for Swin Blocks)
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # Complex implementation omitted for placeholder
        return x # Incorrect, just for structure

class SwinTransformerBlock(nn.Module):
    # Placeholder - Replace with actual implementation (including shifted window logic)
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in [0, window_size)"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # Attention mask calculation for shifted windows omitted for placeholder

    def forward(self, x, x_size):
        # Simplified forward - Needs real implementation of (shifted) window attention
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic Shift (if shift_size > 0) - omitted

        # Partition windows
        x_windows = window_partition(x, self.window_size) # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (pass attn_mask if needed)
        attn_windows = self.attn(x_windows, mask=None) # Placeholder mask=None

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W) # B H' W' C

        # Reverse Cyclic Shift (if shift_size > 0) - omitted

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    """ Placeholder - A basic Swin Transformer layer for one stage. """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution # Should be a tuple (H, W) now
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution, # Pass resolution here
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(self.input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):  # x_size 是动态传入的
        H, W = x_size
        # ... (执行 SwinTransformerBlock) ...
        for blk in self.blocks:
            x = blk(x, x_size)  # 把动态 x_size 传给 block

        if self.downsample is not None:
            # 把当前的动态 x_size 传给 downsample
            # 期望 downsample 返回 x_down, (H_down, W_down)
            x_down, (H_down, W_down) = self.downsample(x, x_size)
            # 返回：下采样前的x, 下采样前的H,W, 下采样后的x, 下采样后的H,W
            return x, H, W, x_down, H_down, W_down
        else:
            # 没有下采样，返回两次相同的信息
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Placeholder - Image to Patch Embedding """
    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Allow dynamic input size, calculate patch resolution based on input
        H_patch, W_patch = H // self.patch_size[0], W // self.patch_size[1]
        # assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
        #     f"Input image size ({H}*{W}) is not divisible by patch size ({self.patch_size[0]}*{self.patch_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if isinstance(self.norm, nn.LayerNorm): # Apply norm if it exists
             x = self.norm(x)
        return x, (H_patch, W_patch) # Return patches and their resolution

class PatchMerging(nn.Module):
    """ Placeholder - Patch Merging Layer. """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, x_size): # Make sure forward signature matches BasicLayer call if needed
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        H_new, W_new = H // 2, W // 2
        # Crucially, return the features AND the new resolution tuple
        return x, (H_new, W_new)

class PatchExpand(nn.Module):
    """ Placeholder - Patch Expanding Layer """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim * dim_scale * dim_scale // 2, bias=False) # Typical Swin-Unet expansion
        self.norm = norm_layer(dim // dim_scale) # Norm the target dim

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        x_size: (H, W)
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x) # B, L, C* scale^2 / 2
        C_expanded = x.shape[-1]
        target_C = C // self.dim_scale # Target channel dim

        # Reshape and rearrange
        x = x.view(B, H, W, C_expanded)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=target_C)
        H_new, W_new = H * self.dim_scale, W * self.dim_scale
        x = x.view(B,-1, target_C) # B, (H*scale)*(W*scale), C/scale

        x = self.norm(x)
        return x, H_new, W_new


class FinalPatchExpand_X4(nn.Module):
    """ Placeholder - Final Patch Expand Layer (Upsample 4x) """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
         super().__init__()
         self.input_resolution = input_resolution
         self.dim = dim
         self.expand = nn.Linear(dim, 16 * dim, bias=False) # Expand channels significantly for 4x upsample
         self.norm = norm_layer(dim)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        x = self.expand(x) # B, L, 16*C

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                      h=H, w=W, p1=4, p2=4, c=C) # B, (H*4), (W*4), C
        H_new, W_new = H * 4, W * 4
        x = x.view(B, -1, C) # B, (H*4)*(W*4), C
        x = self.norm(x)
        return x, H_new, W_new


# ==============================================================================
# Swin-Unet Implementation with adapted interface
# ==============================================================================
class swinUnet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1,
                 img_size=256, patch_size=4,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()


        _img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        _patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.num_classes = out_ch
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=_img_size, patch_size=_patch_size, in_chans=in_ch, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # --- Calculate initial patches resolution based on img_size ---
        patches_resolution = self.patch_embed.patches_resolution
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = patches_resolution # Store for reference if needed

        # Absolute position embedding (optional)
        if self.ape:
             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
             nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
             print(f"Initialized APE for {num_patches} patches.")

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build ENCODER (Swin Transformer stages + Patch Merging)
        self.encoder_layers = nn.ModuleList()
        current_dim = embed_dim
        current_resolution = patches_resolution # Start with initial patch resolution
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=current_dim,
                               input_resolution=current_resolution, # <--- PASS RESOLUTION
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.encoder_layers.append(layer)

            # Update dim and resolution for the *next* layer's init
            if i_layer < self.num_layers - 1:
                current_dim = int(current_dim * 2)
                current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)


        # Build BOTTLENECK (A Swin Transformer BasicLayer)
        # Uses the final resolution and dimension from the encoder loop
        self.bottleneck = BasicLayer(dim=self.num_features,
                                input_resolution=current_resolution, # <--- PASS RESOLUTION
                                depth=depths[-1],
                                num_heads=num_heads[-1],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:-1]):sum(depths)],
                                norm_layer=norm_layer,
                                downsample=None, # No downsampling in bottleneck
                                use_checkpoint=use_checkpoint)


        # Build DECODER (Patch Expanding + Swin Transformer stages)
        self.decoder_layers = nn.ModuleList()
        # We need to calculate dimensions and resolutions carefully, going upwards
        for i_layer in reversed(range(self.num_layers)):
            # Resolution *before* upsampling (output of previous decoder or bottleneck)
            res_before_upsample = (patches_resolution[0] // (2**(i_layer+1)),
                                   patches_resolution[1] // (2**(i_layer+1)))
            # Dimension *before* upsampling
            dim_before_upsample = int(embed_dim * 2**(i_layer+1))

            # Resolution *after* upsampling (matches corresponding encoder stage)
            res_after_upsample = (patches_resolution[0] // (2**i_layer),
                                   patches_resolution[1] // (2**i_layer))
            # Dimension *after* upsampling (half of before)
            dim_after_upsample = int(embed_dim * 2**i_layer)

            # Dimension of skip connection (same as dim_after_upsample)
            dim_skip = dim_after_upsample
            # Dimension after concatenation
            dim_after_concat = dim_after_upsample + dim_skip
            # Dimension output by the decoder stage (same as dim_after_upsample)
            dim_decoder_output = dim_after_upsample


            # Patch Expanding layer first
            upsample_layer = PatchExpand(input_resolution=res_before_upsample, # <--- PASS RESOLUTION
                                         dim=dim_before_upsample,
                                         dim_scale=2, norm_layer=norm_layer)

            # Layer to project concatenated features
            concat_projection = nn.Linear(dim_after_concat, dim_decoder_output)

            # BasicLayer for decoder stage
            decoder_stage = BasicLayer(dim=dim_decoder_output, # Use projected dim
                                       input_resolution=res_after_upsample, # <--- PASS RESOLUTION
                                       depth=depths[i_layer],
                                       num_heads=num_heads[i_layer],
                                       window_size=window_size,
                                       mlp_ratio=self.mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop_rate, attn_drop=attn_drop_rate,
                                       drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                       norm_layer=norm_layer,
                                       downsample=None,
                                       use_checkpoint=use_checkpoint)


            self.decoder_layers.append(nn.ModuleDict({
                'upsample': upsample_layer,
                'concat_proj': concat_projection,
                'stage': decoder_stage
            }))


        # Final Upsampling and Output Layer
        # Input resolution is the initial patches_resolution
        self.final_upsample = FinalPatchExpand_X4(input_resolution=patches_resolution, # <--- PASS RESOLUTION
                                                  dim=embed_dim, norm_layer=norm_layer)
        self.output_conv = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # --- forward method remains largely the same as the previous version ---
    # It handles dynamic input size via padding/cropping and passes dynamic x_size
    def forward(self, x):
        B, C, H, W = x.shape
        H_orig, W_orig = H, W

        _patch_size = self.patch_embed.patch_size
        pad_h = (_patch_size[0] - H % _patch_size[0]) % _patch_size[0]
        pad_w = (_patch_size[1] - W % _patch_size[1]) % _patch_size[1]
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[-2], x.shape[-1] # Update H, W to padded size

        # patch_embed's forward now correctly returns dynamic resolution
        x, current_resolution = self.patch_embed(x) # B, L, C & (Ph, Pw)

        if self.ape:
             # APE interpolation logic might be needed here if using dynamic input sizes
             if x.size(1) != self.absolute_pos_embed.size(1):
                 print(f"Warning: Input patch count {x.size(1)} differs from APE init size {self.absolute_pos_embed.size(1)}. Interpolation needed.")
                 # Add interpolation here if required
                 pass
             x = x + self.absolute_pos_embed[:,:x.size(1),:]


        x = self.pos_drop(x)

        skip_connections = {} # Use dict to store skips with resolution

        # Encoder Path
        for i, layer in enumerate(self.encoder_layers):
            # ... (存储 skip connection) ...
            skip_x, skip_H, skip_W = x, current_resolution[0], current_resolution[1]
            skip_connections[i] = (skip_x, (skip_H, skip_W))

            # BasicLayer 现在返回 x_pre, H_pre, W_pre, x_post, H_post, W_post
            x_pre_downsample, H_pre, W_pre, x_post_downsample, H_post, W_post = layer(x, current_resolution)

            # 将当前张量 x 更新为下采样*之后*的张量
            x = x_post_downsample
            # 将 current_resolution 更新为下采样*之后*的分辨率
            current_resolution = (H_post, W_post)  # <-- 使用 BasicLayer 返回的下采样后的 H, W


        # Bottleneck
        x, H_bottle, W_bottle, _, _, _ = self.bottleneck(x, current_resolution)
        current_resolution = (H_bottle, W_bottle)


        # Decoder Path
        for i, decoder_module in enumerate(self.decoder_layers):
            skip_idx = self.num_layers - 1 - i
            skip, skip_res = skip_connections[skip_idx]
            upsample_layer = decoder_module['upsample']
            concat_proj = decoder_module['concat_proj']
            decoder_stage = decoder_module['stage']

            # Pass current dynamic resolution to upsample
            x, H_up, W_up = upsample_layer(x, current_resolution)
            current_resolution = (H_up, W_up)

            assert current_resolution == skip_res, \
                f"Resolution mismatch in decoder {i}: Upsampled {current_resolution}, Skip {skip_res}"

            x = torch.cat([skip, x], dim=-1)
            x = concat_proj(x)

            # Pass current dynamic resolution to decoder stage
            x, H_stage, W_stage, _, _, _ = decoder_stage(x, current_resolution)
            current_resolution = (H_stage, W_stage)


        # Final Upsampling and Output Projection
        # Pass current dynamic resolution to final upsample
        x, H_final, W_final = self.final_upsample(x, current_resolution)
        current_resolution = (H_final, W_final)

        x = x.transpose(-1, -2)
        final_H, final_W = current_resolution
        x = x.view(B, self.embed_dim, final_H, final_W)

        out = self.output_conv(x)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H_orig, :W_orig]

        return out
