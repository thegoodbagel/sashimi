from torch.utils.data import Dataset
from PIL import Image
import os

class SushiDataset(Dataset):
    def __init__(self, df, img_dir, label_column="species", transform=None):
        self.df = df
        self.img_dir = img_dir
        self.label_column = label_column
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row[self.label_column]   # use dynamic column here
        return image, label
