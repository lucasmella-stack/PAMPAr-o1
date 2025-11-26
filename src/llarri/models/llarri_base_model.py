import pytorch_lightning as pl
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional

from .encoder_vit import ViTEncoder, ViTEncoderConfig
from .decoder_trocr import TrOCRDecoder, TrOCRDecoderConfig

# Simple character-level vocabulary
VOCAB = (
    list("abcdefghijklmnopqrstuvwxyz")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("0123456789")
    + list(".,!?:;\"'()-")
    + ["<pad>", "<sos>", "<eos>"]
)
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
ID2CHAR = {i: c for i, c in enumerate(VOCAB)}

def tokenize(text: str) -> List[int]:
    """Convert a string into a list of token IDs.
    Unknown characters are mapped to the <pad> token.
    """
    return [CHAR2ID.get(ch, CHAR2ID["<pad>"]) for ch in text]

def detokenize(ids: List[int]) -> str:
    """Convert a list of token IDs back into a string, ignoring special tokens."""
    special = {CHAR2ID["<pad>"], CHAR2ID["<sos>"], CHAR2ID["<eos>"]}
    chars = [ID2CHAR.get(i, "") for i in ids if i not in special]
    return "".join(chars)

class LlarriBaseModel(pl.LightningModule):
    def __init__(
        self,
        encoder_cfg: Optional[ViTEncoderConfig] = None,
        decoder_cfg: Optional[TrOCRDecoderConfig] = None,
        vocab: Optional[List[str]] = None,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg or ViTEncoderConfig()
        self.decoder_cfg = decoder_cfg or TrOCRDecoderConfig()
        self.encoder = ViTEncoder(self.encoder_cfg)
        self.decoder = TrOCRDecoder(self.decoder_cfg)
        # Vocabulary handling
        if vocab is not None:
            self.vocab = vocab
            self.char2id = {c: i for i, c in enumerate(vocab)}
            self.id2char = {i: c for i, c in enumerate(vocab)}
        else:
            self.vocab = VOCAB
            self.char2id = CHAR2ID
            self.id2char = ID2CHAR

    def forward(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Run encoder and decoder.
        ``images``: (B, C, H, W)
        ``labels``: optional token IDs for teacher forcing.
        """
        encoder_hidden = self.encoder(images)
        decoder_output = self.decoder(encoder_hidden, labels=labels)
        return decoder_output

    def generate(self, images: torch.Tensor, max_length: int = 128, **gen_kwargs):
        """Auto‑regressive generation.
        Returns a list of decoded strings (one per batch element).
        """
        encoder_hidden = self.encoder(images)
        generated_ids = self.decoder.generate(
            encoder_hidden_states=encoder_hidden, max_length=max_length, **gen_kwargs
        )
        # generated_ids shape: (B, seq_len)
        return [detokenize(ids.tolist()) for ids in generated_ids]

    def export_onnx(self, sample_image: torch.Tensor, onnx_path: str):
        """Export the full model (encoder + decoder) to ONNX.
        ``sample_image`` should be a single image tensor (C, H, W).
        """
        self.eval()
        dummy_input = sample_image.unsqueeze(0)  # (1, C, H, W)
        torch.onnx.export(
            self,
            dummy_input,
            onnx_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=14,
        )

    # Placeholder Lightning methods – users can extend as needed
    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self(images, labels=targets)
        loss = output.loss if hasattr(output, "loss") else torch.tensor(0.0)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        output = self(images, labels=targets)
        loss = output.loss if hasattr(output, "loss") else torch.tensor(0.0)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
