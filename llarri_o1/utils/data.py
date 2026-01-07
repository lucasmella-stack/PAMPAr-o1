# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Carga de datos para entrenamiento.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_mnist_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Obtiene DataLoaders para MNIST.
    
    Args:
        batch_size: Tamaño del batch
        data_dir: Directorio para descargar/cargar datos
        num_workers: Número de workers para carga paralela
        
    Returns:
        Tupla (train_loader, test_loader)
    
    Example:
        >>> train_loader, test_loader = get_mnist_loaders(batch_size=32)
        >>> for images, labels in train_loader:
        ...     # images.shape = (32, 784)
        ...     # labels.shape = (32,)
        ...     pass
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten a 784
    ])
    
    train_dataset = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader
