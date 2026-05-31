
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class UTKFaceDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.image_paths = []

        for f in os.listdir(root_dir):

            if f.endswith(".jpg"):

                try:

                    age = int(f.split("_")[0])

                    if age <= 18:

                        self.image_paths.append(
                            os.path.join(root_dir,f)
                        )

                except:
                    pass

        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):

        img_path = self.image_paths[idx]

        filename = os.path.basename(img_path)

        age = int(filename.split("_")[0])

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image, age
