import os
import sys
import torch
from dataclasses import dataclass

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from llarri.models.encoder_vit import ViTEncoder, ViTEncoderConfig

def test_encoder():
    print("Initializing ViT Encoder...")
    # Use a smaller model for testing if possible, or just the default
    config = ViTEncoderConfig(pretrained_model_name="google/vit-base-patch16-224-in21k", freeze=True)
    encoder = ViTEncoder(config)
    
    print(f"Model hidden size: {encoder.hidden_size}")
    
    # Create dummy input: (Batch, Channels, Height, Width)
    # ViT expects 224x224 usually
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("Forward pass...")
    output = encoder(dummy_input)
    
    print("Output shape:", output.shape)
    
    # Expected output: (Batch, SeqLen, HiddenSize)
    # SeqLen for 224x224 patch16 is (224/16)*(224/16) + 1 (cls token) = 14*14 + 1 = 197
    expected_seq_len = (224 // 16) ** 2 + 1
    assert output.shape == (2, expected_seq_len, encoder.hidden_size)
    
    print("Verification successful!")

if __name__ == "__main__":
    test_encoder()
