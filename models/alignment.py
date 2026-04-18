import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlignment(nn.Module):

    def __init__(self, d_model, temperature=0.1, fusion_type="concat"):
        super().__init__()
        self.temp = temperature
        self.fusion_type = fusion_type
        
        self.graph_proj = nn.Linear(d_model, d_model)
        self.ts_proj = nn.Linear(d_model, d_model)

        if fusion_type == "concat":
            self.fusion = nn.Linear(d_model * 2, d_model)
        elif fusion_type == "add":
            self.fusion = nn.Identity()
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

    def forward(self, h_graph, h_ts, return_loss=False):
        B, N, d = h_graph.shape

        h_g = self.graph_proj(h_graph)  # [B, N, d]
        h_t = self.ts_proj(h_ts)        # [B, N, d]

        h_g = F.normalize(h_g, dim=-1)
        h_t = F.normalize(h_t, dim=-1)
   
        align_loss = 0.0
        if return_loss:

            sim_matrix = torch.matmul(h_g, h_t.transpose(1, 2)) / self.temp  # [B, N, N]
            labels = torch.arange(N, device=h_g.device).unsqueeze(0).repeat(B, 1)  # [B, N]

            loss_g = F.cross_entropy(sim_matrix, labels)
            loss_t = F.cross_entropy(sim_matrix.transpose(1, 2), labels)
            align_loss = (loss_g + loss_t) / 2
        

        if self.fusion_type == "concat":
            h_fused = torch.cat([h_g, h_t], dim=-1)
            h_fused = self.fusion(h_fused)
        elif self.fusion_type == "add":
            h_fused = h_g + h_t
        elif self.fusion_type == "gated":
            gate = self.gate(torch.cat([h_g, h_t], dim=-1))
            h_fused = gate * h_g + (1 - gate) * h_t
        
        if return_loss:
            return h_fused, align_loss
        return h_fused