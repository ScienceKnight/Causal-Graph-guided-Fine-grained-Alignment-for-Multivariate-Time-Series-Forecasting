import torch
import torch.nn as nn
from model.layers import GraphSAGE

class GraphEncoder(nn.Module):
    def __init__(self, var_num, d_model):
        super().__init__()
        self.var_emb = nn.Embedding(var_num, d_model)
        self.sage1 = GraphSAGE(d_model, d_model)
        self.sage2 = GraphSAGE(d_model, d_model)

    def forward(self, adj):
        B = adj.size(0)
        var_idx = torch.arange(self.var_emb.num_embeddings).to(adj.device)
        x = self.var_emb(var_idx).unsqueeze(0).repeat(B, 1, 1)

        x = self.sage1(x, adj)
        x = self.sage2(x, adj)
        return x  