"""
test_training_emnist.py - Prueba rápida de entrenamiento con datos sintéticos

Este script genera palabras sintéticas concatenando caracteres EMNIST
para probar que el pipeline de entrenamiento funciona correctamente.

Uso:
    cd llarri-01
    source venv/bin/activate
    python scripts/test_training_emnist.py
"""

import os
import sys
import random
import string
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class SyntheticWordDataset(Dataset):
    """
    Dataset que genera palabras sintéticas concatenando caracteres EMNIST.
    Útil para probar el pipeline de entrenamiento sin necesidad de
    descargar datasets grandes como IAM.
    """
    
    # Mapeo de índices EMNIST byclass a caracteres
    EMNIST_CLASSES = (
        list(string.digits) +      # 0-9
        list(string.ascii_uppercase) +  # A-Z
        list(string.ascii_lowercase)    # a-z
    )
    
    def __init__(
        self,
        emnist_root: str = "data/external",
        num_samples: int = 1000,
        min_word_len: int = 3,
        max_word_len: int = 8,
        image_height: int = 64,
        train: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            emnist_root: Directorio con datos EMNIST
            num_samples: Número de palabras sintéticas a generar
            min_word_len: Longitud mínima de palabra
            max_word_len: Longitud máxima de palabra
            image_height: Altura de imagen de salida
            train: Si usar split de entrenamiento
            seed: Semilla para reproducibilidad
        """
        self.num_samples = num_samples
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.image_height = image_height
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Cargar EMNIST
        print(f"Cargando EMNIST {'train' if train else 'test'}...")
        self.emnist = datasets.EMNIST(
            root=emnist_root,
            split="byclass",
            train=train,
            download=True,
        )
        
        # Organizar caracteres por clase
        print("Organizando caracteres por clase...")
        self.char_images = {i: [] for i in range(62)}  # 10 dígitos + 26 upper + 26 lower
        
        for idx in range(len(self.emnist)):
            img, label = self.emnist[idx]
            self.char_images[label].append(img)
        
        # Pre-generar palabras y labels
        print(f"Generando {num_samples} palabras sintéticas...")
        self.samples = []
        
        for _ in range(num_samples):
            word_len = random.randint(min_word_len, max_word_len)
            
            # Generar palabra aleatoria (solo letras minúsculas para simplicidad)
            word = ''.join(random.choices(string.ascii_lowercase, k=word_len))
            
            # Crear imagen concatenando caracteres
            char_imgs = []
            for char in word:
                # Obtener índice en EMNIST (a-z están en índices 36-61)
                char_idx = 36 + ord(char) - ord('a')
                
                if self.char_images[char_idx]:
                    # Seleccionar imagen aleatoria de ese carácter
                    char_img = random.choice(self.char_images[char_idx])
                    char_imgs.append(np.array(char_img))
            
            if char_imgs:
                # Concatenar horizontalmente
                word_img = np.concatenate(char_imgs, axis=1)
                self.samples.append((word_img, word))
        
        print(f"Dataset listo: {len(self.samples)} muestras")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_array, text = self.samples[idx]
        
        # Convertir a PIL y redimensionar
        img = Image.fromarray(img_array)
        
        # Redimensionar manteniendo aspect ratio
        aspect = img.width / img.height
        new_width = int(self.image_height * aspect)
        img = img.resize((new_width, self.image_height), Image.Resampling.LANCZOS)
        
        # Convertir a tensor y normalizar
        img_tensor = transforms.ToTensor()(img)
        
        # Expandir a 3 canales si es necesario
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        
        return {
            "image": img_tensor,
            "text": text,
        }


def collate_fn(batch):
    """Función de collate para manejar imágenes de diferente ancho."""
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Pad imágenes al mismo ancho
    max_width = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        # Pad con blanco (1.0)
        pad_width = max_width - img.shape[2]
        if pad_width > 0:
            padding = torch.ones(3, img.shape[1], pad_width)
            img = torch.cat([img, padding], dim=2)
        padded_images.append(img)
    
    return {
        "image": torch.stack(padded_images),
        "text": texts,
    }


def test_dataset():
    """Prueba que el dataset funcione correctamente."""
    print("\n" + "="*60)
    print("TEST: Generación de Dataset Sintético")
    print("="*60)
    
    dataset = SyntheticWordDataset(
        num_samples=100,
        min_word_len=3,
        max_word_len=6,
    )
    
    # Verificar algunas muestras
    print("\nMuestras de ejemplo:")
    for i in range(5):
        sample = dataset[i]
        print(f"  {i}: '{sample['text']}' - shape: {sample['image'].shape}")
    
    # Probar DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch shape: {batch['image'].shape}")
    print(f"Textos: {batch['text']}")
    
    return dataset


def test_model_forward():
    """Prueba un forward pass del modelo."""
    print("\n" + "="*60)
    print("TEST: Forward Pass del Modelo")
    print("="*60)
    
    from llarri.models.llarri_base_model import LlarriBaseModel
    
    # Crear modelo con configuración por defecto
    print("Creando modelo...")
    model = LlarriBaseModel(
        learning_rate=1e-4,
    )
    
    # Mover a GPU si disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    model = model.to(device)
    
    # Crear input dummy
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    print("Ejecutando forward pass...")
    model.eval()
    with torch.no_grad():
        # Generar texto
        generated_texts = model.generate(dummy_input, max_length=32)
        print(f"Textos generados: {generated_texts}")
    
    print("✅ Forward pass exitoso!")
    return model


def test_training_step():
    """Prueba un paso de entrenamiento."""
    print("\n" + "="*60)
    print("TEST: Paso de Entrenamiento")
    print("="*60)
    
    from llarri.models.llarri_base_model import LlarriBaseModel
    
    # Crear modelo
    model = LlarriBaseModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    
    # Crear batch dummy con el formato esperado
    batch_size = 2
    seq_len = 32
    
    # pixel_values: imágenes 224x224
    pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # labels: tokens del texto (usar tokenizer del modelo)
    dummy_texts = ["hello", "world"]
    labels = model.tokenizer(
        dummy_texts, 
        padding="max_length", 
        max_length=seq_len, 
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    print(f"Input shape: {pixel_values.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Textos: {dummy_texts}")
    
    # Training step
    print("Ejecutando training_step...")
    batch = {
        "pixel_values": pixel_values,
        "labels": labels,
    }
    loss = model.training_step(batch, batch_idx=0)
    
    print(f"Loss: {loss.item():.4f}")
    print("✅ Training step exitoso!")
    
    return loss


def run_mini_training():
    """Ejecuta un mini-entrenamiento de prueba."""
    print("\n" + "="*60)
    print("TEST: Mini-Entrenamiento (3 epochs)")
    print("="*60)
    
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from llarri.models.llarri_base_model import LlarriBaseModel
    
    # Dataset
    train_dataset = SyntheticWordDataset(num_samples=50, train=True, seed=42)
    val_dataset = SyntheticWordDataset(num_samples=20, train=False, seed=123)
    
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, 
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Modelo
    model = LlarriBaseModel(
        learning_rate=1e-4,
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        precision="16-mixed",
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        val_check_interval=1.0,
        default_root_dir="outputs/test_run",
    )
    
    # Entrenar
    print("\nIniciando entrenamiento...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✅ Mini-entrenamiento completado!")
    print(f"Mejor val_loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")


def main():
    print("="*60)
    print("PRUEBA DE PIPELINE DE ENTRENAMIENTO - LlarriOCR")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Test 1: Dataset
        test_dataset()
        
        # Test 2: Forward pass
        test_model_forward()
        
        # Test 3: Training step
        test_training_step()
        
        # Test 4: Mini training
        print("\n¿Ejecutar mini-entrenamiento? (esto tomará ~5 minutos)")
        response = input("Escribe 'si' para continuar: ").strip().lower()
        
        if response in ['si', 'sí', 'yes', 'y', 's']:
            run_mini_training()
        else:
            print("Mini-entrenamiento saltado.")
        
        print("\n" + "="*60)
        print("✅ TODAS LAS PRUEBAS PASARON")
        print("="*60)
        print("\nEl pipeline está listo. Para entrenar con datos reales:")
        print("  python -m llarri.training.train_base --config configs/training.yaml")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
