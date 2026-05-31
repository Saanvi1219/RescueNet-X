
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class HeteroRescueGNN(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        self.case_lin = Linear(3, hidden_dim)
        self.face_lin = Linear(1, hidden_dim)
        self.age_lin = Linear(1, hidden_dim)
        self.traj_lin = Linear(1, hidden_dim)
        self.zone_lin = Linear(2, hidden_dim)

        self.conv1 = HeteroConv({
            ("case", "has_face", "face"): SAGEConv((-1, -1), hidden_dim),
            ("case", "has_age", "age"): SAGEConv((-1, -1), hidden_dim),
            ("case", "has_trajectory", "trajectory"): SAGEConv((-1, -1), hidden_dim),
            ("trajectory", "points_to", "zone"): SAGEConv((-1, -1), hidden_dim),
            ("zone", "near", "zone"): SAGEConv((-1, -1), hidden_dim),
        }, aggr="sum")

        self.zone_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, data):

        x_dict = {
            "case": F.relu(self.case_lin(data["case"].x)),
            "face": F.relu(self.face_lin(data["face"].x)),
            "age": F.relu(self.age_lin(data["age"].x)),
            "trajectory": F.relu(self.traj_lin(data["trajectory"].x)),
            "zone": F.relu(self.zone_lin(data["zone"].x)),
        }

        x_dict = self.conv1(x_dict, data.edge_index_dict)

        zone_logits = self.zone_head(x_dict["zone"]).squeeze(-1)

        return zone_logits
