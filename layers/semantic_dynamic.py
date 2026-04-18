import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticGraphLayer(nn.Module):
    def __init__(self, hidden_dim, adj_dim, dropout=0.1, use_bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adj_dim = adj_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.adj_linear = nn.Linear(adj_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, adj):
        batch_size, seq_len, _ = x.shape
        adj = self.adj_linear(adj).view(batch_size, self.hidden_dim, self.hidden_dim)
        x = self.linear(x)
        x = x.transpose(1, 2)
        x = torch.bmm(adj, x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class DynamicFlowLayer(nn.Module):
    def __init__(self, hidden_dim, window_size=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid()
        )
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=window_size, padding=1, groups=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        if seq_len <= 1:
            return x
        x_prev = x[:, :-1, :]
        x_curr = x[:, 1:, :]
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)[:, 1:, :]
        gate_input = torch.cat([x_prev, x_curr + x_conv], dim=-1)
        gate_input = gate_input.view(-1, hidden_dim * 2)
        g = self.gate(gate_input).view(batch_size, seq_len - 1, hidden_dim)
        out = x_curr * g
        out = self.dropout(out)
        return out

class SemanticDynamicFusion(nn.Module):
    def __init__(self, hidden_dim, adj_dim, dropout=0.1):
        super().__init__()
        self.semantic_layer = SemanticGraphLayer(hidden_dim, adj_dim, dropout)
        self.dynamic_layer = DynamicFlowLayer(hidden_dim, dropout=dropout)
        self.fusion_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        semantic_out = self.semantic_layer(x, adj)
        dynamic_out = self.dynamic_layer(x)
        if semantic_out.shape[1] != dynamic_out.shape[1]:
            semantic_out = semantic_out[:, 1:, :]
        fusion_input = torch.cat([semantic_out, dynamic_out], dim=-1)
        fusion_out = self.fusion_linear(fusion_input)
        fusion_out = self.activation(fusion_out)
        fusion_out = self.dropout(fusion_out)
        fusion_out = self.layer_norm(fusion_out + dynamic_out)
        return fusion_out

    def activation(self, x):
        return F.gelu(x)