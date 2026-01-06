import torchvision.datasets as datasets

if __name__ == "__main__":
    datasets.MNIST(root="data/external", download=True)
