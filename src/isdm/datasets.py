from typing import Sequence
import torch
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    def __init__(self, X, y_lists: Sequence[Sequence[int]]):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = [torch.tensor(y, dtype=torch.long) for y in y_lists]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def collate_multilabel(batch, num_classes: int):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)

    y = torch.zeros((len(batch), num_classes), dtype=torch.float32)
    for i, labels in enumerate(ys):
        if labels.numel() > 0:
            y[i, labels] = 1.0

    return x, y