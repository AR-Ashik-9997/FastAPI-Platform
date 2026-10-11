import numpy as np
import pandas as pd
from fastapi import HTTPException
from app.core.ml_loader import ModelManager
from app.modules.inference.schemas import (
    Income_Salary_InferenceRequest,
    HousePriceInferenceRequest,
    ClassificationResponse,
    RegressionResponse,
)


def run_classification_inference(
    payload: Income_Salary_InferenceRequest,
) -> ClassificationResponse:
    model = ModelManager.get_model("income_salary_classification_model")
    if model is None:
        raise HTTPException(
            status_code=500, detail="income_salary_classification_model not found."
        )

    if isinstance(payload.inputs, dict):
        input_array = pd.DataFrame([payload.inputs])
    else:
        input_array = pd.DataFrame(payload.inputs)

    # Prediction
    prediction = model.predict(input_array)

    # Confidence calculation
    confidence = [f"{round(1.0 * 100, 2)}%"] * len(prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_array)
        confidence = [
            f"{round(float(np.max(prob)) * 100, 2)}%" for prob in probabilities
        ]

    formatted_preds = [p.item() if isinstance(p, np.generic) else p for p in prediction]

    if len(formatted_preds) == 1:
        return ClassificationResponse(
            prediction=formatted_preds[0], confidence=confidence[0]
        )

    return ClassificationResponse(prediction=formatted_preds, confidence=confidence)


def run_regression_inference(payload: HousePriceInferenceRequest) -> RegressionResponse:
    model = ModelManager.get_model("house_price_regression_model")
    if model is None:
        raise HTTPException(
            status_code=500, detail="house_price_regression_model not found."
        )

    if isinstance(payload.inputs, dict):
        input_array = pd.DataFrame([payload.inputs])
    else:
        input_array = pd.DataFrame(payload.inputs)

    # Prediction
    prediction = model.predict(input_array)

    # Confidence calculation
    confidence = None
    if hasattr(model, "predict"):
        try:
            _, std_devs = model.predict(input_array, return_std=True)
            confidence = [float(s) for s in std_devs]
        except Exception:
            confidence = None
    formatted_preds = [float(p.item()) if hasattr(p, "item") else p for p in prediction]
    if len(formatted_preds) == 1:
        return RegressionResponse(
            prediction=formatted_preds[0],
            confidence=confidence[0] if confidence is not None else None,
        )

    return RegressionResponse(prediction=formatted_preds, confidence=confidence)
