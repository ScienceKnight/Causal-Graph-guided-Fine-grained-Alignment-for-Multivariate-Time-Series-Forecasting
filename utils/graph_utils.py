import torch
import numpy as np

def build_adj_matrix(feat_dim, semantic_type):
    adj = np.zeros((feat_dim, feat_dim))
    if semantic_type == 'ett':
        adj[0:2,0:2] = 1
        adj[2:4,2:4] = 1
        adj[4:6,4:6] = 1
        adj[:,-1] = 1
        adj[-1,:] = 1
    elif semantic_type == 'weather':
        adj[:,:] = 1
    elif semantic_type == 'ili':
        adj[0:3,0:3] = 1
        adj[3:5,3:5] = 1
        adj[:,-1] = 1
        adj[-1,:] = 1
    adj = adj - np.eye(feat_dim)
    return torch.FloatTensor(adj)

def normalize_adj(adj):
    degree = torch.sum(adj, dim=1)
    degree_inv = torch.diag(1.0 / (degree + 1e-8))
    adj_norm = torch.matmul(degree_inv, adj)
    return adj_norm