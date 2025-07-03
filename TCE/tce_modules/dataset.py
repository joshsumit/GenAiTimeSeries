import torch
from torch.utils.data import Dataset

# Unlabeled dataset for pretraining
class ETTh1UnlabeledDataset(Dataset):
    def __init__(self, data, input_len=96):
        self.data = data
        self.input_len = input_len

    def __len__(self):
        return len(self.data) - self.input_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_len]
        return torch.tensor(x, dtype=torch.float32)

# Labeled dataset for forecasting
class ForecastingDataset(Dataset):
    def __init__(self, data, input_len, pred_len, target_col):
        self.data = data
        self.input_len = input_len
        self.pred_len = pred_len
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.input_len - self.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_len]
        y = self.data[idx + self.input_len:idx + self.input_len + self.pred_len, self.target_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
