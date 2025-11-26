import torch.nn as nn
from transformers import ViTModel

class ViTEncoder(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)

    def forward(self, x):
        return self.vit(x).last_hidden_state
