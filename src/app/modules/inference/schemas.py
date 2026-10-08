from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Income_Salary_InferenceRequest(BaseModel):
    inputs: list[Dict[str, Any]] = Field(
        ...,
        description="Feature array for Income Salary Model",
        json_schema_extra={
            "example": [
                {
                    "age": 30,
                    "workclass": "Private",
                    "fnlwgt": 200000,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Exec-managerial",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    )


class HousePriceInferenceRequest(BaseModel):
    inputs: List[Dict[str, Any]] = Field(
        ...,
        description="Feature array for House Price Regression model",
        json_schema_extra={
            "example": [
                {
                    "Square_Footage": 2100,
                    "Num_Bedrooms": 3,
                    "Num_Bathrooms": 2.5,
                    "Stories": 2,
                    "Year_Built": 2015,
                    "Garage_Size": 2,
                    "Lot_Size": 6500,
                    "Property_Type": "Single-Family",
                    "Neighborhood_Quality": 8,
                    "Has_Pool": 1,
                    "Has_Fireplace": 1,
                    "Condition": "Excellent",
                }
            ]
        },
    )


class ClassificationResponse(BaseModel):
    prediction: Union[str, int, List[Union[str, int]]] = Field(
        ..., description="Predicted class label or list of labels"
    )
    confidence: Union[str, float, List[Union[str, float]]] = Field(
        ..., description="Confidence probability score(s) or percentage string(s)"
    )


class RegressionResponse(BaseModel):
    prediction: Union[float, List[float]] = Field(
        ..., description="Predicted continuous numeric Values"
    )

    confidence: Optional[Union[float, List[float]]] = Field(
        ..., description="Margine of error or Standard deviation"
    )


class CNNInferenceResponse(BaseModel):
    predicted_class: int = Field(..., description="Predicted class index")
    class_label: Optional[str] = Field(default=None, description="Human readable label")
    confidence: float = Field(..., description="Probability")
    all_probabilities: List[float] = Field(
        default=None, description="Softmax probabilities for all classes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": 0,
                "class_label": "Apple",
                "confidence": 0.965,
                "all_probabilities": [0.965, 0.025, 0.010],
            }
        }
