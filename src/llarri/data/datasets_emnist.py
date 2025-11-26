from torch.utils.data import Dataset
import torchvision.datasets as datasets

class EMNISTDataset(Dataset):
    def __init__(self, root_dir, split="byclass", train=True, transforms=None):
        self.dataset = datasets.EMNIST(root=root_dir, split=split, train=train, download=True, transform=transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
