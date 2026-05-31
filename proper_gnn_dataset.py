
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

class ProperTrajectoryGraphDataset(Dataset):

    def __init__(self, plt_files, load_trajectory, kmeans, max_graphs=3000, seq_len=100):

        self.graphs = []
        count = 0

        for file in plt_files:

            if count >= max_graphs:
                break

            try:
                traj = load_trajectory(file)
                coords = traj[["lat", "lon"]].values.astype(np.float32)

                if len(coords) <= seq_len + 1:
                    continue

                # take first seq_len points as graph nodes
                seq = coords[:seq_len]

                # target = zone of next point
                next_point = coords[seq_len].reshape(1, -1)
                label = int(kmeans.predict(next_point)[0])

                # node features: lat, lon, normalized time
                coords_tensor = torch.tensor(seq, dtype=torch.float32)
                time = torch.linspace(0, 1, seq_len).view(-1, 1)
                x = torch.cat([coords_tensor, time], dim=1)

                # temporal edges
                edges = []
                for i in range(seq_len - 1):
                    edges.append([i, i + 1])
                    edges.append([i + 1, i])

                edge_index = torch.tensor(edges, dtype=torch.long).t()

                graph = Data(
                    x=x,
                    edge_index=edge_index,
                    y=torch.tensor(label, dtype=torch.long)
                )

                self.graphs.append(graph)
                count += 1

            except:
                continue

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
