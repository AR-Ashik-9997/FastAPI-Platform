from pydantic import BaseModel, Field
from typing import List

class InferenceRequest(BaseModel):
    inputs: List[float] = Field(..., description="Feature array for AI/ML/DL model", example=[5.1, 3.5, 1.4, 0.2])

class InferenceResponse(BaseModel):
    prediction: int
    confidence: float