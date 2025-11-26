import pytorch_lightning as pl
import torch.nn as nn
from .encoder_vit import ViTEncoder
from .decoder_trocr import TrOCRDecoder

class LlarriBaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = ViTEncoder()
        self.decoder = TrOCRDecoder()

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        return decoder_outputs

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
