import torch.nn as nn
from dataclasses import dataclass
from transformers import TrOCRForCausalLM

@dataclass
class TrOCRDecoderConfig:
    """Configuration for the TrOCR decoder.

    Attributes
    ----------
    pretrained_model_name: str
        Name of the HuggingFace model to load.
    freeze: bool, default=False
        Whether to freeze the decoder parameters during training.
    """
    pretrained_model_name: str = "microsoft/trocr-base-handwritten"
    freeze: bool = False

class TrOCRDecoder(nn.Module):
    """Wrapper around HuggingFace's ``TrOCRForCausalLM``.

    Provides a ``forward`` method compatible with the encoder output and a
    ``generate`` method for auto‑regressive inference.
    """

    def __init__(self, config: TrOCRDecoderConfig = TrOCRDecoderConfig()):
        super().__init__()
        self.config = config
        self.decoder = TrOCRForCausalLM.from_pretrained(config.pretrained_model_name)
        if config.freeze:
            for p in self.decoder.parameters():
                p.requires_grad = False

    def forward(self, encoder_hidden_states, labels=None):
        """Pass encoder hidden states (and optional labels) to the decoder.

        Parameters
        ----------
        encoder_hidden_states: torch.Tensor
            Shape ``(batch, seq_len, hidden_dim)``.
        labels: torch.Tensor, optional
            Token IDs for teacher‑forcing during training.
        """
        return self.decoder(encoder_hidden_states=encoder_hidden_states, labels=labels)

    def generate(self, encoder_hidden_states, max_length=128, **gen_kwargs):
        """Auto‑regressive generation using the underlying ``generate``.

        Returns the generated token IDs tensor.
        """
        return self.decoder.generate(encoder_hidden_states=encoder_hidden_states, max_length=max_length, **gen_kwargs)

