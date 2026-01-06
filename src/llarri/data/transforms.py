import torchvision.transforms as T
import torch
from typing import Optional

class AddGaussianNoise:
    """Add Gaussian noise to tensor."""
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

def get_transforms(stage="train", img_height=128, img_width=512, normalize=True):
    """
    Get transform pipeline for training or validation.
    
    Args:
        stage: "train" or "val"
        img_height: Target height for resizing (default 128 for line-level)
        img_width: Target width for resizing (default 512 for line-level)
        normalize: Whether to apply normalization
    
    Returns:
        Composed torchvision transforms
    """
    if stage == "train":
        transforms = [
            # Convert to grayscale if needed (keep 3 channels for ViT compatibility)
            T.Grayscale(num_output_channels=3),
            # Resize to target dimensions
            T.Resize((img_height, img_width)),
            # Random rotation ±10 degrees
            T.RandomRotation(degrees=10, fill=255),
            # Random affine (shear and stretch)
            T.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=10,
                fill=255
            ),
            # Convert to tensor
            T.ToTensor(),
            # Random brightness and contrast
            T.ColorJitter(brightness=0.3, contrast=0.3),
            # Add Gaussian noise
            AddGaussianNoise(mean=0.0, std=0.05),
        ]
        
        if normalize:
            # ImageNet normalization (for ViT pretrained models)
            transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        return T.Compose(transforms)
    
    else:  # validation
        transforms = [
            # Convert to grayscale (keep 3 channels for ViT)
            T.Grayscale(num_output_channels=3),
            # Resize to target dimensions
            T.Resize((img_height, img_width)),
            # Convert to tensor
            T.ToTensor(),
        ]
        
        if normalize:
            # ImageNet normalization
            transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        return T.Compose(transforms)

# Export convenience functions
def train_transforms(img_height=128, img_width=512, normalize=True):
    """Get training transforms with aggressive augmentations."""
    return get_transforms("train", img_height, img_width, normalize)

def val_transforms(img_height=128, img_width=512, normalize=True):
    """Get validation transforms without augmentations."""
    return get_transforms("val", img_height, img_width, normalize)
