import torch
import os

def save_model(model, path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def save_result(result, path):
    with open(path, 'w') as f:
        f.write(str(result))