
import torch
import torch.nn as nn

class LSTMTrajectoryPredictor(nn.Module):

    def __init__(
        self,
        input_dim=2,
        hidden_dim=128,
        num_layers=3,
        num_zones=8
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(

            nn.Linear(hidden_dim,64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64,num_zones)
        )

    def forward(self,x):

        out,(h,c) = self.lstm(x)

        final_hidden = h[-1]

        return self.fc(final_hidden)
