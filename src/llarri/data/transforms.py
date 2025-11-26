import torchvision.transforms as T

def get_transforms(stage="train"):
    if stage == "train":
        return T.Compose([
            T.ToTensor(),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
        ])
