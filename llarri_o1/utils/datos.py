"""
LLARRI-O1 - Cargador de Datos
==============================

Funciones para cargar y preprocesar datasets.

Autor: Lucas Mella (Segunda Cabeza)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, Dict, Any
import os


def cargar_mnist(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Carga el dataset MNIST.
    
    Args:
        root: Directorio donde guardar los datos
        batch_size: Tamaño del batch
        num_workers: Workers para cargar datos
        pin_memory: Si usar pinned memory (más rápido para GPU)
        download: Si descargar si no existe
    
    Returns:
        (train_loader, test_loader)
    """
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def crear_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders a partir de datasets.
    
    Args:
        train_dataset: Dataset de entrenamiento
        test_dataset: Dataset de prueba
        batch_size: Tamaño del batch
        num_workers: Workers para cargar datos
        pin_memory: Si usar pinned memory
    
    Returns:
        (train_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def get_dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """
    Obtiene información de un dataset.
    
    Args:
        dataset: Dataset de PyTorch
    
    Returns:
        Diccionario con información del dataset
    """
    info = {
        "longitud": len(dataset),
        "tipo": type(dataset).__name__
    }
    
    # Intentar obtener forma de una muestra
    try:
        sample = dataset[0]
        if isinstance(sample, tuple):
            x, y = sample
            info["forma_entrada"] = list(x.shape) if hasattr(x, 'shape') else None
            info["tipo_etiqueta"] = type(y).__name__
        else:
            info["forma_entrada"] = list(sample.shape) if hasattr(sample, 'shape') else None
    except:
        pass
    
    return info


class FlattenTransform:
    """Transforma una imagen a vector plano"""
    def __call__(self, x):
        return x.view(-1)


def cargar_mnist_plano(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    download: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Carga MNIST con las imágenes aplanadas a vectores de 784 dimensiones.
    
    Ideal para LLARRI-O1 que espera entrada de 784 dims.
    """
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        FlattenTransform()
    ])
    
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=download,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader


# ==============================================================================
# DEMO
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("DEMO: Cargador de Datos MNIST")
    print("="*50)
    
    train_loader, test_loader = cargar_mnist_plano(batch_size=64)
    
    print(f"\nDataset de entrenamiento:")
    print(f"  Batches: {len(train_loader)}")
    print(f"  Samples: {len(train_loader.dataset)}")
    
    print(f"\nDataset de prueba:")
    print(f"  Batches: {len(test_loader)}")
    print(f"  Samples: {len(test_loader.dataset)}")
    
    # Info de una muestra
    for x, y in train_loader:
        print(f"\nMuestra:")
        print(f"  Input shape: {x.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Input range: [{x.min():.2f}, {x.max():.2f}]")
        break
