import pytorch_lightning as pl
from ..models.llarri_base_model import LlarriBaseModel
from ..data.datamodule_base import LlarriDataModule

def train(config):
    model = LlarriBaseModel(config)
    datamodule = LlarriDataModule(config)
    trainer = pl.Trainer()
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    pass
