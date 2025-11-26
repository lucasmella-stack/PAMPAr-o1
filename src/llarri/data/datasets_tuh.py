from torch.utils.data import Dataset
import os

class TUHDataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return {}
