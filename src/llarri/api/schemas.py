from pydantic import BaseModel

class PredictionRequest(BaseModel):
    image_base64: str

class PredictionResponse(BaseModel):
    text: str
    confidence: float
