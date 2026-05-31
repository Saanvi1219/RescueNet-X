
import torch.nn as nn

class RescueZonePredictor(nn.Module):

    def __init__(self, num_zones=8):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(),

            nn.Linear(64,128),
            nn.ReLU(),

            nn.Linear(128,num_zones)
        )

    def forward(self,x):
        return self.net(x)
