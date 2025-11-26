import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

from llarri.data.transforms import get_transforms

class JSONLDataset(Dataset):
    def __init__(self, jsonl_path: str, transform=None, root_dir: Optional[str] = None):
        self.data = []
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        
        if self.root_dir:
            full_path = self.root_dir / image_path
        else:
            full_path = Path(image_path)
            
        # Load image
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a dummy tensor or handle error appropriately
            # For now, let's raise to be explicit
            raise e

        if self.transform:
            image = self.transform(image)

        return {
            "id": item.get('id', str(idx)),
            "image": image,
            "text": item.get('text', "")
        }

class LlarriDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_path: str, 
        val_path: str, 
        batch_size: int = 32, 
        num_workers: int = 4, 
        data_root: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.config = config or {}
        
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def prepare_data(self):
        # Check if files exist
        if not os.path.exists(self.train_path):
            print(f"Warning: Train file not found at {self.train_path}")
        if not os.path.exists(self.val_path):
            print(f"Warning: Val file not found at {self.val_path}")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = get_transforms("train")
            val_transforms = get_transforms("val")
            
            if os.path.exists(self.train_path):
                self.train_dataset = JSONLDataset(
                    self.train_path, 
                    transform=train_transforms, 
                    root_dir=self.data_root
                )
            
            if os.path.exists(self.val_path):
                self.val_dataset = JSONLDataset(
                    self.val_path, 
                    transform=val_transforms, 
                    root_dir=self.data_root
                )

    def train_dataloader(self):
        if self.train_dataset:
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers,
                pin_memory=True
            )
        return None

    def test_dataloader(self):
        # Using val dataset for test if no test path provided for now
        if self.val_dataset:
             return DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers
            )
        return None
