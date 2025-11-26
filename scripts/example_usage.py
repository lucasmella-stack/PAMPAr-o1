#!/usr/bin/env python3
"""
Example script showing how to use the LlarriBaseModel with corrected implementation.

This demonstrates:
1. How to initialize the model with tokenizer
2. How to setup the DataModule with collate_fn
3. How to train the model with PyTorch Lightning
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytorch_lightning as pl
from transformers import TrOCRProcessor

from llarri.models.llarri_base_model import LlarriBaseModel
from llarri.models.encoder_vit import ViTEncoderConfig
from llarri.models.decoder_trocr import TrOCRDecoderConfig
from llarri.data.datamodule_base import LlarriDataModule


def main():
    """Main training example."""
    
    # Configuration
    TRAIN_JSONL = "data/splits/train.jsonl"
    VAL_JSONL = "data/splits/val.jsonl"
    DATA_ROOT = "data/processed"
    
    # Model configuration
    encoder_config = ViTEncoderConfig(
        pretrained_model_name="google/vit-base-patch16-224-in21k",
        freeze=False  # Set to True to freeze encoder during training
    )
    
    decoder_config = TrOCRDecoderConfig(
        pretrained_model_name="microsoft/trocr-base-handwritten",
        freeze=False  # Set to True to freeze decoder during training
    )
    
    # Initialize model
    print("Initializing model...")
    model = LlarriBaseModel(
        encoder_cfg=encoder_config,
        decoder_cfg=decoder_config,
        learning_rate=5e-5,
    )
    
    # Get tokenizer from model for DataModule
    tokenizer = model.tokenizer
    
    # Initialize DataModule with tokenizer
    print("Setting up DataModule...")
    datamodule = LlarriDataModule(
        train_path=TRAIN_JSONL,
        val_path=VAL_JSONL,
        data_root=DATA_ROOT,
        batch_size=8,
        num_workers=4,
        tokenizer=tokenizer,  # Pass tokenizer for collate_fn
        max_length=128,
        img_height=128,
        img_width=512,
    )
    
    # Setup PyTorch Lightning Trainer
    print("Configuring trainer...")
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",  # Uses GPU if available
        devices=1,
        precision="16-mixed",  # Mixed precision training
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints",
                filename="llarri-{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, datamodule)
    
    # Save final model
    print("Saving final model...")
    trainer.save_checkpoint("checkpoints/llarri-final.ckpt")
    
    print("Training complete!")


def inference_example():
    """Example of using trained model for inference."""
    import torch
    from PIL import Image
    from llarri.data.transforms import val_transforms
    
    # Load trained model
    print("Loading trained model...")
    model = LlarriBaseModel.load_from_checkpoint(
        "checkpoints/llarri-final.ckpt"
    )
    model.eval()
    
    # Load and preprocess image
    image_path = "data/user_samples/sample.png"
    transform = val_transforms(img_height=128, img_width=512)
    
    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Generate prediction
    with torch.no_grad():
        predictions = model.generate(
            pixel_values=pixel_values,
            max_length=128,
            num_beams=4,
        )
    
    print(f"Predicted text: {predictions[0]}")


def export_onnx_example():
    """Example of exporting model to ONNX format."""
    import torch
    from PIL import Image
    from llarri.data.transforms import val_transforms
    
    # Load trained model
    print("Loading model for ONNX export...")
    model = LlarriBaseModel.load_from_checkpoint(
        "checkpoints/llarri-final.ckpt"
    )
    model.eval()
    
    # Create sample input
    transform = val_transforms(img_height=128, img_width=512)
    sample_image = torch.randn(3, 128, 512)  # Dummy image
    
    # Export to ONNX
    print("Exporting to ONNX...")
    model.export_onnx(
        sample_image=sample_image,
        onnx_path="models/llarri_base.onnx"
    )
    
    print("ONNX export complete!")


if __name__ == "__main__":
    # Run training
    main()
    
    # Uncomment to run inference example
    # inference_example()
    
    # Uncomment to export to ONNX
    # export_onnx_example()
