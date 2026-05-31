
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

class TrajectoryGraphDataset(Dataset):

    def __init__(self, seq_dataset, max_graphs=5000):

        self.graphs = []

        limit = min(len(seq_dataset), max_graphs)

        for i in range(limit):

            seq, label = seq_dataset[i]

            # seq shape: [100, 2]
            coords = seq.float()

            # Node features:
            # lat, lon, normalized_time
            time = torch.linspace(0, 1, coords.shape[0]).view(-1, 1)
            x = torch.cat([coords, time], dim=1)

            # Edges: connect every GPS point to next point
            edges = []

            for j in range(coords.shape[0] - 1):
                edges.append([j, j + 1])
                edges.append([j + 1, j])

            edge_index = torch.tensor(edges, dtype=torch.long).t()

            graph = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(label, dtype=torch.long)
            )

            self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
