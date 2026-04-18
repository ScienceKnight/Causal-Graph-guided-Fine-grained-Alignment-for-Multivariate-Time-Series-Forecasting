import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, use_bias=True, norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.norm = norm
        self.w = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        if self.norm:
            self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        support = self.w(x)
        out = torch.bmm(adj, support)
        if self.norm:
            out = self.layer_norm(out)
        out = self.dropout(out)
        out = F.gelu(out)
        return out

class GATConv(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=1, dropout=0.1, use_bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.w = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.a = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        batch_size, seq_len, _ = x.shape
        x = self.w(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        x = x.transpose(1, 2)
        a_input = torch.cat([x.unsqueeze(3).repeat(1, 1, 1, seq_len, 1), x.unsqueeze(2).repeat(1, 1, seq_len, 1, 1)], dim=-1)
        a_input = a_input.view(batch_size, self.n_heads, seq_len * seq_len, 2 * self.head_dim)
        e = self.leaky_relu(self.a(a_input).squeeze(-1))
        e = e.view(batch_size, self.n_heads, seq_len, seq_len)
        adj = adj.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        e = e.masked_fill(adj == 0, -1e9)
        attn = F.softmax(e, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, x)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.out_dim)
        return out

class GraphConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, adj_dim, n_heads=1, dropout=0.1):
        super().__init__()
        self.graph_conv = GATConv(in_dim, out_dim, n_heads, dropout)
        self.semantic_adj = nn.Linear(adj_dim, out_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        residual = x
        adj = self.semantic_adj(adj)
        adj = F.softmax(adj, dim=-1)
        out = self.graph_conv(x, adj)
        out = self.dropout(out)
        out = self.layer_norm1(out + residual)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = self.layer_norm2(out + residual)
        return out