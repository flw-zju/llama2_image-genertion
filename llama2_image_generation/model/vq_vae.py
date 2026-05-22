"""
modified from https://github.com/CompVis/latent-diffusion/
"""

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import einsum
from einops import rearrange


def nonlinearity(x):
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups, in_channels, eps=1e-8, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        pad = (0,1,0,1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = Normalize(channels)
        self.proj_in = nn.Conv2d(channels, channels, 3, 1, 1)
        self.query = nn.Conv2d(channels, channels, 1, 1, 0)
        self.key = nn.Conv2d(channels, channels, 1, 1, 0)
        self.value = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):

        x = self.norm(x)
        x = self.proj_in(x)
        b, c, h, w = x.shape
        q = self.query(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # [B, HW, C//8]
        k = self.key(x).view(b, self.num_heads, self.head_dim, h * w)  # [B, C//8, HW]
        v = self.value(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # [B, C, HW]

        attn = torch.matmul(q, k) / self.head_dim ** 0.5  # 缩放点积注意力
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).permute(0, 1, 3, 2)

        out = out.reshape(b, c, h, w)

        return self.proj(out) + x


class Encoder(nn.Module):
    def __init__(self, *, ch=128, ch_mult=(1,2,2,4),
                 in_channels=3,
                  z_channels=64, double_z=False,
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.Sequential()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            block.append(ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        ))
            block.append(ResnetBlock(in_channels=block_out,
                                     out_channels=block_out,
                                     ))
            block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in)

            self.down.append(down)
        self.mid = nn.Sequential(
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in,
                        ),
            SpatialSelfAttention(block_in),
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in,
                        ),
        )
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):

            h = self.down[i_level].block[0](h)
            h = self.down[i_level].block[1](h)

            if i_level != self.num_resolutions-1:

                h = self.down[i_level].downsample(h)

        h = self.mid(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch=128, out_ch=3, ch_mult=(1,2,2,4),
                 z_channels=64,
                 tanh_out=False
                 ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)

        self.tanh_out = tanh_out

        block_in = ch*ch_mult[self.num_resolutions-1]


        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.mid = nn.Sequential(
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in,
                        ),
            SpatialSelfAttention(block_in),
            ResnetBlock(in_channels=block_in,
                        out_channels=block_in,
                        ),
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.Sequential()

            block_out = ch*ch_mult[i_level]
            block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                        ))
            block.append(ResnetBlock(in_channels=block_out,
                                     out_channels=block_out,
                                     ))

            block_in = block_out

            up = nn.Module()
            up.block = block

            if i_level != 0:
                up.upsample = Upsample(block_in)

            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        h = self.conv_in(z)
        h = self.mid(h)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):

            h = self.up[i_level].block[0](h)
            h = self.up[i_level].block[1](h)
            #h = self.up[i_level].block[2](h)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class EMAVectorQuantizer2(nn.Module):
    def __init__(self, n_e, e_dim, beta, gamma=0.99, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.gamma = gamma  # EMA 动量
        self.legacy = legacy

        # 码本
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0/self.n_e, 1.0/self.n_e)

        # EMA 核心：cluster size & embedding sum
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer('embed_sum', self.embedding.weight.clone())

    def forward(self, z):

        # 维度变换
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if self.training:
            indices_one_hot = torch.zeros(min_encoding_indices.shape[0], self.n_e, device=z.device)
            indices_one_hot.scatter_(1, min_encoding_indices.unsqueeze(1), 1.0)

            # 更新 cluster count
            cluster_size = indices_one_hot.sum(0)
            self.cluster_size.data.mul_(self.gamma).add_(cluster_size, alpha=1 - self.gamma)

            # 更新 embedding 总和
            embed_sum = torch.matmul(indices_one_hot.t(), z_flattened)
            self.embed_sum.data.mul_(self.gamma).add_(embed_sum, alpha=1 - self.gamma)

            # 平滑 & 更新码本
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.n_e * 1e-5) * n
            embed_normalized = self.embed_sum / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)

        # 损失函数
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # 梯度直通
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, (None, None, min_encoding_indices)


class VQModel(nn.Module):
    def __init__(self,
                 n_embed=2048,
                 embed_dim=64,
                 image_key="image",
                 lr_g_factor=1.0,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantize = EMAVectorQuantizer2(n_embed, embed_dim, 0.25)
        self.quant_conv = torch.nn.Conv2d(64, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 64, 1)

        self.lr_g_factor = lr_g_factor

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)

        quant, emb_loss, info = self.quantize(h)

        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)

        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def training_step(self, x):

        xrec, qloss, ind = self(x, return_pred_indices=True)

        return xrec, qloss, ind

    def configure_optimizers(self):
        lr_g = self.lr_g_factor*2e-4
        print("lr_g", lr_g)
        codebook_params = list(self.quantize.embedding.parameters())
        other_params = (list(self.encoder.parameters()) +
                        list(self.decoder.parameters()) +
                        list(self.quant_conv.parameters()) +
                        list(self.post_quant_conv.parameters()))

        opt_ae = torch.optim.Adam([
            {'params': other_params, 'lr': lr_g},  # 原有学习率
            {'params': codebook_params, 'lr': lr_g}  # 码本学习率翻倍
        ])

        return opt_ae



