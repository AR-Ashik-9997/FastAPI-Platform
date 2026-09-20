from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

class InferenceRequest(BaseModel):
    inputs: List[Dict[str, Any]] = Field(..., description="Feature array for AI/ML/DL model", json_schema_extra={
            "example": [
                {
                    "age": 30,
                    "workclass": "Private",
                    "hours-per-week": 40
                }
            ]
        }
    )

class ClassificationResponse(BaseModel):
    prediction: Union[str, int, List[Union[str, int]]] = Field(
        ..., 
        description="Predicted class label or list of labels"
    )
    confidence: Union[float, List[float]] = Field(
        ..., 
        description="Confidence probability score(s)"
    )

class RegressionResponse(BaseModel):
    prediction: Union[float, List[float]] = Field(
        ..., 
        description="Predicted continuous numeric Values"
    )

    confidence: Optional[Union[float, List[float]]] = Field(
        ..., 
        description="Margine of error or Standard deviation"
    )
