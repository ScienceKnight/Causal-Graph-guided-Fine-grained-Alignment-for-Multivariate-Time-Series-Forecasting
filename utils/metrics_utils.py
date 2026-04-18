import numpy as np

def calculate_metrics(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    mape = np.mean(np.abs((pred - true)) / (np.abs(true) + 1e-8))
    smape = np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
    return mse, mae, mape, smape

def calculate_classification_metrics(pred, true):
    acc = np.mean(pred == true)
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return acc, precision, recall, f1