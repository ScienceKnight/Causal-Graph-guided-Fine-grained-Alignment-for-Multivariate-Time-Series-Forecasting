import torch
import numpy as np

def generate_random_mask(shape, missing_rate):
    mask = np.random.rand(*shape) < missing_rate
    return torch.FloatTensor(mask)

def apply_mask(x, mask, fill_value=0.0):
    x_masked = x.clone()
    x_masked[mask.bool()] = fill_value
    return x_masked

def generate_block_mask(shape, block_len, missing_rate):
    batch, seq_len, dim = shape
    mask = np.zeros((batch, seq_len, dim))
    block_num = int(seq_len * missing_rate / block_len)
    for b in range(batch):
        for _ in range(block_num):
            start = np.random.randint(0, seq_len - block_len)
            mask[b, start:start+block_len, :] = 1
    return torch.FloatTensor(mask)