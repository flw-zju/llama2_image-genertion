import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tqdm import tqdm


def get_1d_rotary_pos_embed(dim, pos, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))
    t = torch.arange(0, pos).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def broadcast_rope(freqs_cis, x):
    # x:[bs, sl, n_head, head_dim]
    n_dim = x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    new_shape = [d if i==1 or i==n_dim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*new_shape)


def apply_rope(freqs_cis, q, k):
    # q,k:[bs, sl, n_head, head_dim]
    q_out = torch.view_as_complex(q.view(*q.shape[:-1], -1, 2))
    k_out = torch.view_as_complex(k.view(*k.shape[:-1], -1, 2))
    freqs_cis = broadcast_rope(freqs_cis, q_out)
    q_out = torch.view_as_real(q_out * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_out * freqs_cis).flatten(3)
    return q_out, k_out


def repeat_kv(x, n_rep):
    # x:[bs, sl, n_head, head_dim]
    bs, sl, n_head, head_dim = x.shape
    return (x[:, :, :, None, :]
            .expand(bs, sl, n_head, n_rep, head_dim)
            .reshape(bs, sl, n_head*n_rep, head_dim))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x / (
                torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True)) + self.eps)
                * self.weight)


class Attention(nn.Module):
    def __init__(self, dim, head_dim, n_rep,
                 max_bs, max_sl, dropout=0.1,
                 mode="train"):
        super().__init__()
        assert dim % head_dim == 0
        self.dim = dim
        self.head_dim = head_dim
        self.n_rep = n_rep
        self.n_q_head = dim // head_dim
        self.n_kv_head = self.n_q_head // n_rep
        self.mode = mode

        self.q = nn.Linear(dim, self.n_q_head * head_dim*2, bias=True)
        self.k = nn.Linear(dim, self.n_kv_head * head_dim, bias=True)
        self.v = nn.Linear(dim, self.n_kv_head * head_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.n_q_head * head_dim, dim, bias=True)

        if mode == "eval":
            self.k_cache = torch.zeros(max_bs, max_sl, self.n_kv_head, head_dim)
            self.v_cache = torch.zeros(max_bs, max_sl, self.n_kv_head, head_dim)

    def forward(self, x, freqs_cis, mask=None, start_pos=None):
        bs, sl, _ = x.shape
        q_gate = self.q(x).view(bs, sl, self.n_kv_head, -1)
        q, gate = torch.split(q_gate, [self.head_dim * self.n_rep,
                                                              self.head_dim * self.n_rep], dim=-1)
        gate = gate.reshape(bs, sl, -1, self.head_dim)
        q = q.reshape(bs, sl, -1, self.head_dim)


        k = self.k(x).view(bs, sl, self.n_kv_head, self.head_dim)
        v = self.v(x).view(bs, sl, self.n_kv_head, self.head_dim)

        q, k = apply_rope(freqs_cis, q, k)
        if start_pos is not None and self.mode == "eval":
            self.k_cache[:bs, start_pos:start_pos+sl] = k
            self.v_cache[:bs, start_pos:start_pos+sl] = v

            ks = self.k_cache[:bs, :start_pos+sl].to(x.device)
            vs = self.v_cache[:bs, :start_pos+sl].to(x.device)
        else:
            ks = k
            vs = v

        ks = repeat_kv(ks, self.n_rep)
        vs = repeat_kv(vs, self.n_rep)

        q = q.transpose(1, 2)
        ks = ks.transpose(1, 2)
        vs = vs.transpose(1, 2)

        scores = torch.matmul(q, ks.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float("-inf"))
        scores = F.softmax(scores, dim=-1)

        scores = self.dropout(scores)
        out = torch.matmul(scores, vs)
        out = out.transpose(1, 2).contiguous()
        out = out * torch.sigmoid(gate)
        out = out.view(bs, sl, -1)
        out = self.out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, dropout=0.1):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.fc0 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.fc1(x)) * self.fc0(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, head_dim, n_rep,
                 max_bs, max_sl,
                 hidden_dim, multiple_of,
                 dropout=0.1, mode="train"):
        super().__init__()
        self.norm0 = RMSNorm(dim)
        self.attn = Attention(dim, head_dim, n_rep,
                              max_bs, max_sl, dropout,
                              mode)
        self.norm1 = RMSNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, multiple_of, dropout)

    def forward(self, x, start_pos, freqs_cis, mask=None):
        x = x + self.attn(self.norm0(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.norm1(x))
        return x


class Llama2(nn.Module):
    def __init__(self, vocab_sz=2048,
                 label_sz=7, dim=512,
                 head_dim=64, n_rep=2,
                 max_bs=64, max_sl=256,
                 multiple_of=512, depth=16,
                 mlp_ratio=4,
                 dropout=0.,
                 mode="train"):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.vocab_sz = vocab_sz
        self.label_sz = label_sz

        self.token_embedding = nn.Embedding(vocab_sz, dim)
        self.label_embedding = nn.Embedding(label_sz, dim)
        self.sos_emb = nn.Parameter(torch.randn(1, dim))

        self.decoder = nn.ModuleList([])
        for _ in range(depth):
            self.decoder.append(
                TransformerBlock(
                    dim, head_dim, n_rep,
                    max_bs, max_sl,
                    hidden_dim, multiple_of,
                    dropout, mode))

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_sz, bias=False)

        freqs_cis = get_1d_rotary_pos_embed(head_dim, max_sl*2)
        self.register_buffer("freqs_cis", freqs_cis)

        self.apply(self._init_weights)
        nn.init.normal_(self.output.weight, mean=0.0, std=1 / math.sqrt(dim))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / math.sqrt(module.in_features))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1 / math.sqrt(module.embedding_dim))

    def forward(self, x, start_pos=None):
        bs, sl, _ = x.shape

        if start_pos is not None:
            freqs = self.freqs_cis[start_pos:start_pos + sl]
        else:
            freqs = self.freqs_cis[:sl]

        mask = None
        if sl > 1:
            mask = torch.ones((sl, sl)).to(x.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        for i, block in enumerate(self.decoder):
            x = block(x, freqs, mask, start_pos)

        x = self.norm(x)
        logits = self.output(x)
        return logits

    def train_step(self, x, l):
        logits = self(x[:, :-1], l)
        logits = logits[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            x.reshape(-1).long(),
        )
        return loss

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, batch_size, device, top_k=5):
        l = torch.randint(0, self.label_sz, (batch_size,)).to(device)
        y = self.label_embedding(l).unsqueeze(1)
        x = self.sos_emb.unsqueeze(0).repeat(y.shape[0], 1, 1)
        x = torch.cat([y, x], dim=1)

        ans = []
        for i in tqdm(range(256)):
            start_pos = i + 1 if i > 0 else 0
            logits = self(x, start_pos)
            logits = logits[:, -1, :]
            logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            x = self.token_embedding(ix)
            ans.append(ix)
        return torch.cat(ans, dim=1)












