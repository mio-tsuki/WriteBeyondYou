import torch
from torch.utils.data import Dataset


class StrokeDataset(Dataset):
    def __init__(self, sequences: list[torch.Tensor]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index]
