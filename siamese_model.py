
import torch
import torch.nn as nn
from face_encoder import FaceEncoder

class SiameseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = FaceEncoder()

    def forward(self, img1, img2):

        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)

        return emb1, emb2
