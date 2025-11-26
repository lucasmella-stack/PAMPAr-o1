import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import TrOCRForCausalLM, TrOCRConfig

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
        
        # Get special token IDs from config
        decoder_config = self.decoder.config
        self.pad_token_id = decoder_config.pad_token_id if decoder_config.pad_token_id is not None else 1
        self.decoder_start_token_id = decoder_config.decoder_start_token_id if decoder_config.decoder_start_token_id is not None else 2
        
        if config.freeze:
            for p in self.decoder.parameters():
                p.requires_grad = False

    def _shift_tokens_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Shift input ids one token to the right for teacher forcing.
        Prepends decoder_start_token_id.
        """
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = self.decoder_start_token_id
        
        # Replace pad tokens that were shifted
        shifted.masked_fill_(shifted == -100, self.pad_token_id)
        
        return shifted

    def forward(self, encoder_hidden_states, labels=None, decoder_input_ids=None):
        """Pass encoder hidden states (and optional labels) to the decoder.

        Parameters
        ----------
        encoder_hidden_states: torch.Tensor
            Shape ``(batch, seq_len, hidden_dim)``.
        labels: torch.Tensor, optional
            Token IDs for computing loss during training.
        decoder_input_ids: torch.Tensor, optional
            Token IDs for decoder input. If None and labels provided, 
            will be created by shifting labels right.
        """
        # Create decoder_input_ids from labels if not provided
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_tokens_right(labels)
        
        # TrOCRForCausalLM uses 'input_ids' not 'decoder_input_ids'
        return self.decoder(
            encoder_hidden_states=encoder_hidden_states,
            input_ids=decoder_input_ids,
            labels=labels
        )

    def generate(self, encoder_hidden_states, max_length=128, **gen_kwargs):
        """Auto‑regressive generation using the underlying ``generate``.

        Returns the generated token IDs tensor.
        """
        return self.decoder.generate(
            encoder_hidden_states=encoder_hidden_states, 
            max_length=max_length, 
            **gen_kwargs
        )

