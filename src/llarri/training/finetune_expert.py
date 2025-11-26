import pytorch_lightning as pl
from ..models.llarri_base_model import LlarriBaseModel

def finetune(base_model_path, config):
    model = LlarriBaseModel.load_from_checkpoint(base_model_path)
    # Adjust model for finetuning
    trainer = pl.Trainer()
    # trainer.fit(...)

if __name__ == "__main__":
    pass
