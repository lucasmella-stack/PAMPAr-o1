import pytorch_lightning as pl
from ..models.selector_style import StyleSelector

def train_selector(config):
    model = StyleSelector(input_dim=768, num_styles=5)
    trainer = pl.Trainer()
    # trainer.fit(...)

if __name__ == "__main__":
    pass
