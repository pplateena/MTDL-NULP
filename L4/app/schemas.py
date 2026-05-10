from pydantic import BaseModel
from typing import List


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    top5: List[dict]


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
