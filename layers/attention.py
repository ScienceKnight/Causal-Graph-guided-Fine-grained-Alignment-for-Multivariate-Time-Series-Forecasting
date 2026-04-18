import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, semantic_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads if d_model % n_heads == 0 else d_model // (n_heads - 1)
        self.semantic_dim = semantic_dim if semantic_dim is not None else d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_sem = nn.Linear(self.semantic_dim, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

    def forward(self, q, k, v, semantic_feat=None, mask=None):
        b, l, d = q.shape
        q = self.w_q(q).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
        if semantic_feat is not None:
            semantic_feat = self.w_sem(semantic_feat).view(b, l, self.n_heads, self.d_k).transpose(1, 2)
            k = k + semantic_feat
            v = v + semantic_feat
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, l, d)
        out = self.out(out)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        residual = q
        q = self.w_q(q).view(q.size(0), q.size(1), self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(k.size(0), k.size(1), self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(v.size(0), v.size(1), self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(q.size(0), q.size(2), -1)
        out = self.out(out)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return out, attn

class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.mask = None

    def _create_causal_mask(self, seq_len, device):
        if self.mask is None or self.mask.size(0) != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            self.mask = mask == 0
        return self.mask

    def forward(self, x):
        mask = self._create_causal_mask(x.size(1), x.device)
        out, attn = self.attention(x, x, x, mask=mask)
        return out, attn