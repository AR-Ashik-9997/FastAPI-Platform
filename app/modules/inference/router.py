from fastapi import APIRouter
from app.modules.inference.schemas import InferenceRequest, InferenceResponse
from app.modules.inference import service

router = APIRouter(prefix="/inference", tags=["AI/ML Inference Engine"])

@router.post("/predict", response_model=InferenceResponse)
def predict(payload: InferenceRequest):
    return service.run_model_inference(payload)