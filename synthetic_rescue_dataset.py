import random
import torch
from torch.utils.data import Dataset


class SyntheticRescueBenchmark(Dataset):
    """
    Conference prototype benchmark.

    This creates synthetic multimodal samples so the full architecture can train immediately.
    Replace this with real data loaders after downloading UTKFace, FG-NET, DeepFashion and GeoLife.

    Each item:
    - missing_face
    - found_face
    - clothing_image
    - text_tokens
    - geo_features
    - rescue graph
    - match label
    - rescue zone label
    """
    def __init__(
        self,
        n=2000,
        image_size=64,
        vocab_size=1000,
        text_len=28,
        geo_dim=10,
        graph_nodes=14,
        graph_node_dim=32,
        num_zones=8,
        seed=7
    ):
        super().__init__()
        self.n = n
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.geo_dim = geo_dim
        self.graph_nodes = graph_nodes
        self.graph_node_dim = graph_node_dim
        self.num_zones = num_zones

        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return self.n

    def _img(self, signal):
        img = torch.randn(3, self.image_size, self.image_size) * 0.30 + signal
        return torch.clamp(img, -1, 1)

    def _graph(self, zone, match):
        x = torch.randn(self.graph_nodes, self.graph_node_dim) * 0.4

        rescue_node = 1 + (zone % (self.graph_nodes - 1))

        x[0] += 0.7 if match else -0.2
        x[rescue_node] += 1.2

        adj = torch.eye(self.graph_nodes)

        adj[0, rescue_node] = 1
        adj[rescue_node, 0] = 1

        for i in range(1, self.graph_nodes - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1

        for _ in range(8):
            a = random.randint(1, self.graph_nodes - 1)
            b = random.randint(1, self.graph_nodes - 1)
            adj[a, b] = 1
            adj[b, a] = 1

        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj = adj / deg
        return x.float(), adj.float()

    def __getitem__(self, idx):
        match = 1 if random.random() > 0.5 else 0
        zone = random.randint(0, self.num_zones - 1)

        identity_signal = random.uniform(-0.4, 0.4)

        missing_face = self._img(identity_signal)

        if match:
            found_face = missing_face + torch.randn_like(missing_face) * 0.07
            clothing_image = self._img(identity_signal + 0.2)

            text_tokens = torch.randint(1, self.vocab_size, (self.text_len,))
            text_tokens[0] = 10 + zone
            text_tokens[1] = 101
            text_tokens[2] = 202

            geo = torch.randn(self.geo_dim) * 0.2
            geo[0] = zone / self.num_zones
            geo[1] = 1.0
            geo[2] = 0.8
        else:
            found_face = self._img(random.uniform(-0.9, 0.9))
            clothing_image = self._img(random.uniform(-0.9, 0.9))

            text_tokens = torch.randint(1, self.vocab_size, (self.text_len,))

            geo = torch.randn(self.geo_dim)
            geo[0] = random.random()
            geo[1] = 0.0
            geo[2] = random.random() * 0.3

        graph_x, graph_adj = self._graph(zone, match)

        return {
            "missing_face": missing_face.float(),
            "found_face": found_face.float(),
            "clothing_image": clothing_image.float(),
            "text_tokens": text_tokens.long(),
            "geo_features": geo.float(),
            "graph_x": graph_x.float(),
            "graph_adj": graph_adj.float(),
            "match_label": torch.tensor(match, dtype=torch.float32),
            "zone_label": torch.tensor(zone, dtype=torch.long),
        }
