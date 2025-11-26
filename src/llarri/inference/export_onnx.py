import torch
from ..models.llarri_base_model import LlarriBaseModel

def export_to_onnx(model_path, output_path):
    model = LlarriBaseModel.load_from_checkpoint(model_path)
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_path)

if __name__ == "__main__":
    pass
