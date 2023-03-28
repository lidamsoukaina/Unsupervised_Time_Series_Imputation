import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from induce_nans import random_mask_tensor


class TSImputationTrainDataset(Dataset):
    """Time series imputation dataset for training"""

    def __init__(self, df, seq_length, missing_ratio):
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.seq_length = seq_length
        self.missing_ratio = missing_ratio

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, index):
        assert len(self.data) >= self.seq_length
        # Get a sequence of data of length seq_length
        target = self.data[index : index + self.seq_length]
        # generate masked input and the mask
        masked_input, mask = random_mask_tensor(target, self.missing_ratio, index)
        return target, masked_input, mask


class TSImputationEvalDataset(Dataset):
    """Time series imputation dataset for testing"""

    def __init__(self, df, df_nan, mask, seq_length):
        self.seq_length = seq_length
        self.data = torch.tensor(df.values, dtype=torch.float32)
        self.data_nan = torch.tensor(df_nan.values, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        if len(self.data) >= self.seq_length:
            return len(self.data) - self.seq_length + 1
        return 1

    def __getitem__(self, index):
        # Get a sequence of data of length seq_length or less
        start = index
        end = index + self.seq_length
        target = self.data[start:end]
        masked_input = self.data_nan[start:end]
        mask = self.mask[start:end]
        # add padding if needed
        if end > len(self.data):
            padding = torch.zeros(self.seq_length - len(target), self.data.shape[1])
            target = torch.cat([target, padding], dim=0)
            masked_input = torch.cat([masked_input, padding], dim=0)
            mask = torch.cat([mask, padding.bool()], dim=0)
        masked_input[torch.isnan(masked_input)] = 0
        return target, masked_input, mask
