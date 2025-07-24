from torch.utils.data import Dataset
from PIL import Image
import os

class SushiDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, species_to_idx=None, part_to_idx=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.species_to_idx = species_to_idx
        self.part_to_idx = part_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        species_label = row["species"]
        part_label = row["part"]

        return image, species_label, part_label
