import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

class DataEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, max_seq_len=1000, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.emb = nn.Linear(in_dim, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.emb(x)
        x = self.pos_emb(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.1, activation='gelu'):
        super().__init__()
        self.l1 = nn.Linear(d_in, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_in)
        self.dropout = nn.Dropout(dropout)
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1, activation='gelu'):
        super().__init__()
        self.feed_forward = FeedForward(d_model, d_hidden, dropout, activation)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x