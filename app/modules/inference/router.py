from fastapi import APIRouter
from app.modules.inference.schemas import InferenceRequest, ClassificationResponse,RegressionResponse
from app.modules.inference import service

router = APIRouter(prefix="/inference", tags=["AI/ML Inference Engine"])

@router.post("/predict/classification", response_model=ClassificationResponse)
def predict_classification(payload: InferenceRequest):
    return service.run_classification_inference(payload)

@router.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(payload: InferenceRequest):
    return service.run_regression_inference(payload)

