
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class BetterRescueGNN(nn.Module):

    def __init__(self, hidden_dim=64):
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
            ("face", "rev_face", "case"): SAGEConv((-1, -1), hidden_dim),
            ("age", "rev_age", "case"): SAGEConv((-1, -1), hidden_dim),
            ("trajectory", "rev_traj", "case"): SAGEConv((-1, -1), hidden_dim),
            ("trajectory", "points_to", "zone"): SAGEConv((-1, -1), hidden_dim),
            ("zone", "related_to", "zone"): SAGEConv((-1, -1), hidden_dim),
        }, aggr="sum")

        self.conv2 = HeteroConv({
            ("trajectory", "points_to", "zone"): SAGEConv((-1, -1), hidden_dim),
            ("zone", "related_to", "zone"): SAGEConv((-1, -1), hidden_dim),
        }, aggr="sum")

        self.zone_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
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
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        zone_logits = self.zone_head(x_dict["zone"]).squeeze(-1)

        return zone_logits
