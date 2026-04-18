import torch
import torch.nn as nn
from model.layers import TransformerBlock

class TemporalEncoder(nn.Module):
    def __init__(self, seq_len, var_num, d_model, n_heads, n_layers):
        super().__init__()
        self.var_num = var_num
        self.seq_len = seq_len
        self.d_model = d_model

        self.proj = nn.Linear(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model*4) for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: [B, T, N]
        x = x.transpose(1, 2)  # [B, N, T]
        x = self.proj(x)       # [B, N, d_model]

        for block in self.blocks:
            x = block(x)
        return x 