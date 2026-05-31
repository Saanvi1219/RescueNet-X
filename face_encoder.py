
import torch.nn as nn
from torchvision.models import resnet18

class FaceEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = resnet18(weights="DEFAULT")

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            128
        )

    def forward(self,x):
        return self.backbone(x)
