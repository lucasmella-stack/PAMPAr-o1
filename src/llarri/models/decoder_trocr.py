import torch.nn as nn
from transformers import TrOCRForCausalLM

class TrOCRDecoder(nn.Module):
    def __init__(self, pretrained_model_name="microsoft/trocr-base-handwritten"):
        super().__init__()
        self.decoder = TrOCRForCausalLM.from_pretrained(pretrained_model_name)

    def forward(self, encoder_hidden_states, labels=None):
        return self.decoder(encoder_hidden_states=encoder_hidden_states, labels=labels)
