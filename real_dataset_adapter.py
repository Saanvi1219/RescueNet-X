"""
Template for replacing synthetic data with real public datasets.

Recommended mapping:

1. UTKFace / FG-NET
   - missing_face
   - found_face
   - age-progressed variation

2. DeepFashion / DeepFashion2 / Market1501
   - clothing_image
   - appearance attributes

3. Missing person structured records
   - text_tokens from age, gender, clothing, scars, last-seen description

4. GeoLife / OpenStreetMap
   - geo_features from lat/lon/time/transport-distance/crowd-density

5. Rescue Knowledge Graph
   - graph_x and graph_adj created from:
     child node
     last seen location node
     nearby transport node
     crowd zone node
     historical case node
     public report node

Expected output dictionary:

{
    "missing_face": Tensor[3,H,W],
    "found_face": Tensor[3,H,W],
    "clothing_image": Tensor[3,H,W],
    "text_tokens": Tensor[text_len],
    "geo_features": Tensor[geo_dim],
    "graph_x": Tensor[num_nodes,node_dim],
    "graph_adj": Tensor[num_nodes,num_nodes],
    "match_label": Tensor[],
    "zone_label": Tensor[]
}
"""

from torch.utils.data import Dataset


class RealRescueDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        raise NotImplementedError(
            "Add parsing logic for UTKFace/FG-NET/DeepFashion/GeoLife here."
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
