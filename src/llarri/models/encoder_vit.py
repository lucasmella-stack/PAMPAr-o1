from dataclasses import dataclass
import torch.nn as nn
from transformers import ViTModel, ViTConfig

@dataclass
class ViTEncoderConfig:
    pretrained_model_name: str = "google/vit-base-patch16-224-in21k"
    freeze: bool = False

class ViTEncoder(nn.Module):
    def __init__(self, config: ViTEncoderConfig):
        super().__init__()
        self.config = config
        self.vit = ViTModel.from_pretrained(config.pretrained_model_name)
        
        if config.freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self):
        return self.vit.config.hidden_size

    def forward(self, x):
        # ViTModel returns BaseModelOutputWithPooling
        # we want last_hidden_state: (batch_size, sequence_length, hidden_size)
        return self.vit(x).last_hidden_state
