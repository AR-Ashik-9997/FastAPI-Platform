import numpy as np
from fastapi import HTTPException
from app.core.ml_loader import ModelManager
from app.modules.inference.schemas import InferenceRequest, InferenceResponse

def run_model_inference(payload: InferenceRequest) -> InferenceResponse:
    model = ModelManager.get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded on server.")
    
    # Preprocessing / Tensor array manipulation
    input_array = np.array(payload.inputs).reshape(1, -1)
    
    # Prediction
    prediction = model.predict(input_array)[0]
    
    # Confidence calculation
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_array)[0]
        confidence = float(np.max(probabilities))
        
    return InferenceResponse(prediction=int(prediction), confidence=confidence)