import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Custom Dataset Definition
class BrainTumorMRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        super().__init__()
        self.paths = df['image_path'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform
        self.classes = sorted(list(df['label'].unique()))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.paths[index]).convert('RGB')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        image = self.load_image(index)
        class_name = self.labels[index]
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)
        return image, class_idx