import joblib
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class ModelManager:
    _models = {}

    @classmethod
    def load_all_models(cls):
        if not MODELS_DIR.exists():
            print(f"Warning: Models directory not found at {MODELS_DIR}")
            return

        for file_path in MODELS_DIR.glob("*.pkl"):
            model_name = file_path.stem
            print(f"Loading AI Model '{model_name}' from {file_path}...")
            cls._models[model_name] = joblib.load(file_path)
        print(f"Successfully loaded models: {list(cls._models.keys())}")

    @classmethod
    def get_model(cls, name: str):
        return cls._models.get(name)
