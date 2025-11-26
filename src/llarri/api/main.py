from fastapi import FastAPI
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    return {"text": "prediction placeholder", "confidence": 0.99}

@app.get("/health")
async def health():
    return {"status": "ok"}
