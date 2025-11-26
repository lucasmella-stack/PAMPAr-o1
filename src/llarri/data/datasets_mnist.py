from torch.utils.data import Dataset
import torchvision.datasets as datasets

class MNISTDataset(Dataset):
    def __init__(self, root_dir, train=True, transforms=None):
        self.dataset = datasets.MNIST(root=root_dir, train=train, download=True, transform=transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
