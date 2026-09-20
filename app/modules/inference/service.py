import io
import torch
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from fastapi import UploadFile,HTTPException
from app.core.ml_loader import ModelManager
from app.modules.inference.schemas import InferenceRequest, ClassificationResponse,RegressionResponse,CNNInferenceResponse

def run_classification_inference(payload: InferenceRequest) -> ClassificationResponse:
    model = ModelManager.get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded on server.")
    
    if isinstance(payload.inputs,dict):
        input_array = pd.DataFrame([payload.inputs])
    else:
        input_array = pd.DataFrame(payload.inputs)
    
    # Prediction
    prediction = model.predict(input_array)
    
    # Confidence calculation
    confidence = 1.0*len(prediction)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_array)
        confidence = [float(np.max(prob)) for prob in probabilities]
    formatted_preds = [
        p.item() if isinstance(p, np.generic) else p 
        for p in prediction
    ]    
    if len(formatted_preds) == 1:
        return ClassificationResponse(
            prediction=formatted_preds[0], 
            confidence=confidence[0]
        )
        
    return ClassificationResponse(
        prediction=formatted_preds, 
        confidence=confidence
    )


def run_regression_inference(payload: InferenceRequest) -> RegressionResponse:
    model = ModelManager.get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded on server.")
    
    if isinstance(payload.inputs,dict):
        input_array = pd.DataFrame([payload.inputs])
    else:
        input_array = pd.DataFrame(payload.inputs)
    
    # Prediction
    prediction = model.predict(input_array)
    
    # Confidence calculation
    confidence = None
    if hasattr(model, "predict"):
        try:
            _,std_devs = model.predict(input_array,return_std=True)
            confidence = [float(s) for s in std_devs]
        except Exception:
            confidence=None    
    formatted_preds = [
        float(p.item()) if hasattr(p, 'item') else p 
        for p in prediction
    ]    
    if len(formatted_preds) == 1:
        return RegressionResponse(
            prediction=formatted_preds[0], 
            confidence=confidence[0] if confidence is not None else None
        )
        
    return RegressionResponse(
        prediction=formatted_preds, 
        confidence=confidence
    )

def image_process(file_bytes:bytes,target_size=(224,224))->np.ndarray:
    try:
        image=Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image=image.resize(target_size)
        img_array=np.array(image,dtype=np.float32)/255.0
        img_array=np.transpose(img_array,(1,0,1))
        img_array=np.expand_dims(img_array,axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing faild:{str(e)}")

def run_cnn_inference(file:UploadFile)->CNNInferenceResponse:
    model = ModelManager.get_model()
    if model is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded on server.")
    
    file_bytes=file.file.read()
    input_tensor=image_process(file_bytes)
    outputs=None
    try:
        if hasattr(model,"eval")and callable(model):
            model.eval()
            with torch.no_grad():
                tensor_input=torch.from_numpy(input_tensor)
                raw_out=model(tensor_input)
                probs=torch.softmax(raw_out,dim=1)
                outputs=probs.numpy()[0]
        elif hasattr(model,"predict"):
            raw_out=model.predict(input_tensor,verbose=0)
            outputs=raw_out[0]
        else:
            raise HTTPException(status_code=500, detail="Unsupported DL model framework.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model execution error: {str(e)}")
    
    predicted_class = int(np.argmax(outputs))
    confidence = float(outputs[predicted_class])
    all_probs = [float(p) for p in outputs]

    return CNNInferenceResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        all_probabilities=all_probs
    )
print("aaaaaaa")
