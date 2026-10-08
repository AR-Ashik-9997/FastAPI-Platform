import joblib
import os

class ModelManager:
    _model = None

    @classmethod
    def load_model(cls, model_path: str = "models/trained_model.pkl"):
        if os.path.exists(model_path):
            print(f"Loading AI Model from {model_path}...")
            cls._model = joblib.load(model_path)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model file not found at {model_path}. Inference endpoints may fail.")

    @classmethod
    def get_model(cls):
        return cls._model