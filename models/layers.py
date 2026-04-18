import torch
import torch.nn as nn
import torch.nn.functional as F

# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# GraphSAGE Layer
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x, adj):
        neighbor = torch.matmul(adj, x)
        out = torch.cat([x, neighbor], dim=-1)
        out = self.linear(out)
        return F.relu(out)

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, graph_feat, temp_feat):
        sim = torch.matmul(graph_feat, temp_feat.T) / self.temp
        labels = torch.arange(sim.size(0)).to(sim.device)
        loss = F.cross_entropy(sim, labels)
        return loss