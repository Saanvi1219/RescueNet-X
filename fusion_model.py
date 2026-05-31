
import torch
import torch.nn as nn

class RescueFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(4,16),
            nn.ReLU(),

            nn.Linear(16,8),
            nn.ReLU(),

            nn.Linear(8,1),

            nn.Sigmoid()
        )

    def forward(self,x):

        return self.network(x)
