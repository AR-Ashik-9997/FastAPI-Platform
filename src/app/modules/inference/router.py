from fastapi import APIRouter,File,UploadFile,HTTPException
from app.modules.inference.schemas import InferenceRequest, ClassificationResponse,RegressionResponse,CNNInferenceResponse
from app.modules.inference import service

router = APIRouter(prefix="/inference", tags=["AI/ML Inference Engine"])

@router.post("/predict/classification", response_model=ClassificationResponse)
def predict_classification(payload: InferenceRequest):
    return service.run_classification_inference(payload)

@router.post("/predict/regression", response_model=RegressionResponse)
def predict_regression(payload: InferenceRequest):
    return service.run_regression_inference(payload)

@router.post("/predict/images", response_model=CNNInferenceResponse)
async def predict_image(file:UploadFile=File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,detail="Upload file must be a valid image formate (JPG, PNG, JPEG).")
    return service.run_cnn_inference(file)

