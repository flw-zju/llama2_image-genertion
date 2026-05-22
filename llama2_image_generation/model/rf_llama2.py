import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vae import AutoencoderKL


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[:dim // 2].float() / dim))
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def get_2d_rope(embed_dim, h, w):
    assert embed_dim % 4 == 0
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)

    emb_h = get_1d_rotary_pos_embed(embed_dim // 2, grid_h)  # (H, D/4)
    emb_w = get_1d_rotary_pos_embed(embed_dim // 2, grid_w)  # (W, D/4)

    # 2D 网格扩展
    emb_h = emb_h.unsqueeze(1).repeat(1, w, 1)  # (H, W, D/4)
    emb_w = emb_w.unsqueeze(0).repeat(h, 1, 1)  # (H, W, D/4)

    # 拼接成复数形式
    freqs_cis = torch.cat([emb_h, emb_w], dim=-1)  # (H, W, D/2)
    freqs_cis = freqs_cis.flatten(0, 1)  # (H*W, D/2)
    return freqs_cis


def broadcast_(freqs, x):
    n_dim = x.ndim
    shape = [d if i==1 or i==n_dim-1 else 1 for i, d in enumerate(x.shape)]
    freqs = freqs.view(*shape)
    return freqs


def apply_rotate(freqs, q, k):
    q = torch.view_as_complex(q.view(*q.shape[:-1], -1, 2))
    k = torch.view_as_complex(k.view(*k.shape[:-1], -1, 2))
    freqs = broadcast_(freqs, q).to(q.device)
    q_out = torch.view_as_real(q * freqs).flatten(3)
    k_out = torch.view_as_real(k * freqs).flatten(3)
    return q_out, k_out


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    bs, sl, n_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, sl, n_heads, n_rep, head_dim)
        .reshape(bs, sl, n_heads*n_rep, head_dim)
    )


class GaussianFourierProjection(nn.Module):
  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class Attention(nn.Module):
    def __init__(self, dim, head_dim=64, n_rep=1):
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.n_q_heads = dim // head_dim
        self.n_kv_heads = self.n_q_heads // n_rep
        self.n_rep = n_rep

        self.q = nn.Linear(dim, self.n_q_heads*self.head_dim, bias=True)
        self.k = nn.Linear(dim, self.n_kv_heads*self.head_dim, bias=True)
        self.v = nn.Linear(dim, self.n_kv_heads*self.head_dim, bias=True)
        self.out = nn.Linear(self.n_q_heads*self.head_dim, dim, bias=True)

    def forward(self, x, freqs_cis):
        bs, sl, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.view(bs, sl, self.n_q_heads, self.head_dim)
        k = k.view(bs, sl, self.n_kv_heads, self.head_dim)
        v = v.view(bs, sl, self.n_kv_heads, self.head_dim)

        q, k = apply_rotate(freqs_cis, q, k)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        score = F.softmax(score, dim=-1)

        out = torch.matmul(score, v)
        out = out.transpose(1, 2).contiguous().view(bs, sl, -1)
        out = self.out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * (
                (hidden_dim + multiple_of - 1) // multiple_of)
        self.w0 = nn.Linear(dim, hidden_dim, bias=True)
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w0(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim,
                 head_dim=64, n_rep=2,
                 multiple_of=256):
        super().__init__()
        self.norm0 = RMSNorm(dim)
        self.attn = Attention(dim, head_dim, n_rep)

        self.norm1 = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, multiple_of)

        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False)
        )

    def forward(self, x, t, freqs_cis):
        t = self.t_proj(t)[:, None, :]
        x = x + self.attn(self.norm0(x) + t, freqs_cis)
        x = x + self.ffn(self.norm1(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.t_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x, t):
        y = self.norm(x)
        t = self.t_proj(t)
        y = y + t[:, None, :]
        y = self.linear(y)
        return y


class Transformer(nn.Module):
    def __init__(self, in_ch=4, in_size=32,
                 patch_size=2,
                 dim=768,
                 head_dim=64, n_rep=2,
                 multiple_of=256, depth=12,
                 t_emb_dim=512):
        super().__init__()
        hidden_dim = dim * 4
        self.dim = dim
        self.patch_sz = patch_size
        self.in_ch = in_ch
        self.num_patches = (in_size // patch_size) ** 2

        self.time_emb = GaussianFourierProjection(t_emb_dim, 16)
        self.time_proj = nn.Sequential(
            nn.Linear(t_emb_dim*2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.patch_conv = nn.Conv2d(in_ch, dim,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    padding=0)

        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(TransformerBlock(
                dim, hidden_dim,
                head_dim, n_rep,
                multiple_of
            ))

        self.final = FinalLayer(dim, patch_size, in_ch)
        freqs_cis = get_2d_rope(head_dim, in_size // patch_size, in_size // patch_size)
        self.register_buffer("freqs_cis", freqs_cis)
        self.freqs_cis.requires_grad_(False)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_conv.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_conv.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_proj[0].weight, std=0.02)
        nn.init.normal_(self.time_proj[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.t_proj.weight, 0)
            nn.init.constant_(block.t_proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final.linear.weight, 0)
        nn.init.constant_(self.final.linear.bias, 0)

    def unpatchify(self, x):
        c = self.in_ch
        p = self.patch_sz
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t):
        t_emb = self.time_emb(torch.log(t))
        t_emb = self.time_proj(t_emb)

        x_emb = (self.patch_conv(x)
                 .view(x.shape[0], self.dim, -1)
                 .transpose(1, 2))
        for module in self.transformer:
            x_emb = module(x_emb, t_emb, self.freqs_cis)
        x_emb = self.final(x_emb, t_emb)
        x_emb = self.unpatchify(x_emb)
        return x_emb


class RectifiedFlow(nn.Module):
    def __init__(self, in_ch=4, in_size=32,
                 patch_size=2,
                 dim=512,
                 head_dim=64, n_rep=1,
                 multiple_of=256, depth=12,
                 t_emb_dim=512,
                 vae_path=None,
                 device=None
                 ):
        super().__init__()
        self.dit = Transformer(in_ch, in_size,
                 patch_size,
                 dim,
                 head_dim, n_rep,
                 multiple_of, depth,
                 t_emb_dim
                 )
        self.vae = AutoencoderKL()
        self.device = device
        self.vae.init_from_ckpt(vae_path)
        self.set_params()

    def set_params(self):
        self.vae.requires_grad_(False)
        self.vae.eval()

    def get_vae_sample(self, m, s):
        noise = torch.randn(m.shape).to(device=self.device)
        return m + s * noise

    def get_train_item(self, x_1, x_0=None, eps=1e-3):
        bs = x_1.shape[0]
        t = torch.rand(bs, device=self.device)
        if x_0 is None:
            x_0 = torch.randn_like(x_1)
        x_t = x_0 * (1 - t.view(-1, 1, 1, 1)) + x_1 * t.view(-1, 1, 1, 1)
        return x_t, t * (1 - eps) + eps, x_1 - x_0

    def flow(self, x_t, t):
        pred = self.dit(x_t, t)
        return pred

    def forward(self, m, s):
        with torch.no_grad():
            x = self.get_vae_sample(m, s)
            x = x * 0.2039
        x_t, t, tar = self.get_train_item(x)
        pred = self.flow(x_t, t*999)
        flow_loss = F.mse_loss(pred, tar)
        return flow_loss



