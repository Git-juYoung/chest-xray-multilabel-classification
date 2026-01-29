import torch

from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, path, target, mask, transform):
        self.path = path
        self.target = target
        self.mask = mask
        self.transform = transform

    def __len__(self):
        
        return len(self.path)

    def __getitem__(self, idx):
        path = self.path[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        target = torch.tensor(self.target[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)
        
        return image, target, mask

