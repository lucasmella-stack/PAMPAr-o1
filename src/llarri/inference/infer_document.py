from .infer_line import LineInference
from .preprocess_opencv import preprocess_image

class DocumentInference:
    def __init__(self, model_path):
        self.line_inference = LineInference(model_path)

    def process_document(self, document_path):
        # 1. Preprocess
        # 2. Detect lines
        # 3. Infer each line
        pass
