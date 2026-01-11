# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza

"""
Carga de datos para entrenamiento.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, List, Optional
from pathlib import Path


# ============================================================================
# CORPUS DE TEXTO
# ============================================================================

class TextDataset(Dataset):
    """Dataset para secuencias de texto tokenizado."""
    
    def __init__(self, tokens: List[int], seq_length: int):
        self.tokens = tokens
        self.seq_length = seq_length
        # Número de secuencias completas que podemos crear
        self.num_sequences = (len(tokens) - 1) // seq_length
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        
        input_ids = torch.tensor(self.tokens[start:end], dtype=torch.long)
        targets = torch.tensor(self.tokens[start+1:end+1], dtype=torch.long)
        
        return input_ids, targets


def cargar_corpus(
    path: Path,
    tokenizer,
    max_tokens: Optional[int] = None
) -> List[int]:
    """
    Cargar y tokenizar un corpus de texto.
    
    Args:
        path: Ruta al archivo de texto
        tokenizer: Tokenizer de SentencePiece
        max_tokens: Máximo número de tokens a cargar (None = todo)
    
    Returns:
        Lista de token IDs
    """
    print(f"📖 Cargando corpus desde {path}...")
    
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   Caracteres: {len(text):,}")
    
    # Tokenizar
    tokens = tokenizer.Encode(text)
    print(f"   Tokens totales: {len(tokens):,}")
    
    # Limitar si es necesario
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        print(f"   Tokens limitados a: {len(tokens):,}")
    
    return tokens


def crear_dataloader(
    tokens: List[int],
    batch_size: int,
    seq_length: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Crear DataLoader para entrenamiento de lenguaje.
    
    Args:
        tokens: Lista de token IDs
        batch_size: Tamaño del batch
        seq_length: Longitud de secuencia
        shuffle: Mezclar datos
        num_workers: Workers para carga paralela
    
    Returns:
        DataLoader configurado
    """
    dataset = TextDataset(tokens, seq_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Evitar batches incompletos
    )


# ============================================================================
# MNIST (Legacy)
# ============================================================================

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
