import torch
import torch.nn as nn
from model.temporal_encoder import TemporalEncoder
from model.graph_encoder import GraphEncoder
from model.alignment import CrossModalAlignment

class CausalAlign(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temp_enc = TemporalEncoder(
            seq_len=config.seq_len,
            var_num=config.var_num,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers
        )
        self.graph_enc = GraphEncoder(
            var_num=config.var_num,
            d_model=config.d_model
        )
        self.alignment = CrossModalAlignment(config.d_model)
        self.pred_head = nn.Linear(config.d_model, config.pred_len)

    def forward(self, x_ts, adj, return_loss=False):
        # 时序特征
        h_ts = self.temp_enc(x_ts)
        # 图特征
        h_graph = self.graph_enc(adj)
        # 对齐 + 融合
        h_fused, align_loss = self.alignment(h_graph, h_ts, return_loss)
        # 预测
        pred = self.pred_head(h_fused).transpose(1, 2)
        if return_loss:
            return pred, align_loss
        return pred