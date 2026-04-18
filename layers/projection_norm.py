import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if self.affine:
            return self.weight * (x - mean) / (std + self.eps) + self.bias
        else:
            return (x - mean) / (std + self.eps)

class BatchNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        return x

class PredictHead(nn.Module):
    def __init__(self, d_in, out_dim, seq_len, pred_len, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.proj1 = nn.Linear(d_in, d_in)
        self.proj2 = nn.Linear(d_in, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.fc = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        batch_size, seq_len, d_in = x.shape
        x = self.proj1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.proj2(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x

class ProjectionBlock(nn.Module):
    def __init__(self, d_in, out_dim, seq_len, pred_len, dropout=0.1):
        super().__init__()
        self.predict_head = PredictHead(d_in, out_dim, seq_len, pred_len, dropout)
        self.norm = LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(d_in, out_dim) if d_in != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x[:, -self.pred_len:, :] if x.shape[1] > self.pred_len else x)
        out = self.predict_head(x)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out