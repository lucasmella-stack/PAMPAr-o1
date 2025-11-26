from pydantic import BaseModel

class LabelRequest(BaseModel):
    image_id: str
    image_path: str

class LabelResponse(BaseModel):
    image_id: str
    text: str
