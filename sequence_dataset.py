
import torch
from torch.utils.data import Dataset

class TrajectorySequenceDataset(Dataset):

    def __init__(self, coords, zones, seq_len=100):

        self.X = []
        self.Y = []

        for i in range(len(coords)-seq_len):

            seq = coords[i:i+seq_len]

            target = zones[i+seq_len]

            self.X.append(seq)
            self.Y.append(target)

        self.X = torch.tensor(
            self.X,
            dtype=torch.float32
        )

        self.Y = torch.tensor(
            self.Y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.Y[idx]
