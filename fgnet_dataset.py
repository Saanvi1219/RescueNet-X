
import os
import re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FGNETDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.images = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith(".jpg")
        ]

        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def parse_file(self, file):

        name = file.split(".")[0]

        match = re.match(
            r"(\d+)A(\d+)[a-zA-Z]?",
            name,
            re.IGNORECASE
        )

        if match is None:
            return -1, -1

        person_id = int(match.group(1))
        age = int(match.group(2))

        return person_id, age

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        file = self.images[idx]

        person_id, age = self.parse_file(file)

        img = Image.open(
            os.path.join(self.root_dir, file)
        ).convert("RGB")

        img = self.transform(img)

        return img, person_id, age
