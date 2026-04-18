import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ETTDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv'):
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.root_path = root_path
        self.data_path = data_path
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw.iloc[:, 1:].values  
        self.data = df_raw.astype(np.float32)

        num_train = int(len(self.data) * 0.7)
        num_val = int(len(self.data) * 0.2)
        num_test = len(self.data) - num_train - num_val
        border1s = [0, num_train - self.seq_len, len(self.data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(self.data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data = self.data[border1:border2]

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), \
               torch.tensor(seq_y, dtype=torch.float32)