import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Lightweight CNN encoder.
    In final experiments, this can be replaced with ResNet50/EfficientNet.
    """
    def __init__(self, out_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return self.proj(x)


class TextEncoder(nn.Module):
    """
    Lightweight text encoder.
    In final experiments, replace with Sentence-BERT embeddings.
    """
    def __init__(self, vocab_size=1000, emb_dim=64, out_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, out_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(out_dim * 2, out_dim)

    def forward(self, tokens):
        x = self.embedding(tokens)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.proj(h)


class GeoEncoder(nn.Module):
    """
    Encodes location/time/mobility numeric features.
    Example features:
    lat_norm, lon_norm, time_of_day, distance_to_transport, crowd_density, historical_case_density.
    """
    def __init__(self, geo_dim=10, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geo_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class GCNLayer(nn.Module):
    """
    Simple Graph Convolution:
    H' = ReLU(AHW)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        h = torch.bmm(adj, x)
        return F.relu(self.w(h))


class RescueGraphEncoder(nn.Module):
    """
    Encodes rescue knowledge graph.
    Nodes can represent child, places, transport hubs, crowd areas, past cases.
    """
    def __init__(self, node_dim=32, hidden_dim=64, out_dim=128):
        super().__init__()
        self.gcn1 = GCNLayer(node_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        return h.mean(dim=1)


class CrossModalAttentionFusion(nn.Module):
    """
    Gives explainable attention over modalities.
    Modalities:
    0 face_pair
    1 clothing
    2 text
    3 geo
    4 graph
    """
    def __init__(self, dim=128, num_modalities=5, fused_dim=256):
        super().__init__()
        self.attn_score = nn.Linear(dim, 1)
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_modalities, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

    def forward(self, modalities):
        x = torch.stack(modalities, dim=1)
        scores = self.attn_score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        weighted = x * weights.unsqueeze(-1)
        fused = self.fusion(weighted.flatten(1))
        return fused, weights


class RescueNetX(nn.Module):
    """
    RescueNet-X:
    Multimodal child matching + graph reasoning + rescue-zone trajectory prediction.
    """
    def __init__(self, vocab_size=1000, geo_dim=10, graph_node_dim=32, num_zones=8):
        super().__init__()
        self.face_encoder = ImageEncoder(128)
        self.clothing_encoder = ImageEncoder(128)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, out_dim=128)
        self.geo_encoder = GeoEncoder(geo_dim=geo_dim, out_dim=128)
        self.graph_encoder = RescueGraphEncoder(node_dim=graph_node_dim, out_dim=128)

        self.fusion = CrossModalAttentionFusion(dim=128, num_modalities=5, fused_dim=256)

        self.match_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )

        self.zone_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_zones)
        )

    def forward(self, batch):
        missing_face = self.face_encoder(batch["missing_face"])
        found_face = self.face_encoder(batch["found_face"])

        face_pair = torch.abs(missing_face - found_face)

        clothing = self.clothing_encoder(batch["clothing_image"])
        text = self.text_encoder(batch["text_tokens"])
        geo = self.geo_encoder(batch["geo_features"])
        graph = self.graph_encoder(batch["graph_x"], batch["graph_adj"])

        fused, modality_weights = self.fusion([face_pair, clothing, text, geo, graph])

        match_logit = self.match_head(fused).squeeze(1)
        zone_logits = self.zone_head(fused)

        return {
            "match_logit": match_logit,
            "zone_logits": zone_logits,
            "modality_weights": modality_weights
        }
