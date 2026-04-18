import numpy as np

def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-8] = 1
    return (data - mean) / std, mean, std

def inverse_standardize(data, mean, std):
    return data * std + mean

def train_test_split(data, split_ratio=0.8):
    split = int(len(data) * split_ratio)
    return data[:split], data[split:]

def create_sliding_window(data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:i+seq_len]
        y = data[i+seq_len:i+seq_len+pred_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)