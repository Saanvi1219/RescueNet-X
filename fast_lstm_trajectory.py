
import torch
import torch.nn as nn

class FastLSTMTrajectoryPredictor(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1, num_zones=20):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_zones)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])
