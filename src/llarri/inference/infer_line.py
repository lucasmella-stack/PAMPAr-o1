import torch
from ..models.llarri_base_model import LlarriBaseModel

class LineInference:
    def __init__(self, model_path):
        self.model = LlarriBaseModel.load_from_checkpoint(model_path)
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor)
        return output
