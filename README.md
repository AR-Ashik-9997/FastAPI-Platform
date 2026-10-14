# FastAPI ML Inference Platform

This project is a lightweight FastAPI application for serving trained ML models over HTTP. It loads scikit-learn-compatible model files from the `models/` directory and exposes inference endpoints for classification and regression tasks.

## Overview

The application is intentionally small and modular:

- FastAPI app entrypoint in `src/app/main.py`
- Central configuration in `src/app/core/config.py`
- Model loading logic in `src/app/core/ml_loader.py`
- Inference API routes in `src/app/modules/inference/`

The current implementation serves two trained models:

- `income_salary_classification_model.pkl`
- `house_price_regression_model.pkl`

These models are loaded at startup and can be used via API requests.

## Tech Stack

- Python 3.10+
- FastAPI
- Pydantic v2
- scikit-learn / joblib
- NumPy
- Pandas
- Uvicorn

## Project Structure

```text
Fast-Api/
├── models/
│   ├── house_price_regression_model.pkl
│   └── income_salary_classification_model.pkl
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── ml_loader.py
│       └── modules/
│           └── inference/
│               ├── __init__.py
│               ├── router.py
│               ├── schemas.py
│               └── service.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── README.md
└── .venv/
```

## Configuration

The project uses `pydantic-settings` to manage app configuration via environment variables and defaults.

Current settings in `src/app/core/config.py`:

```python
class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise AI/ML FastAPI Platform"
    API_V1_STR: str = "/api/v1"
```

You can override these values via environment variables if needed.

## Installation

1. Clone the repository and move into the project directory.
2. Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Optional local package install:

```bash
pip install -e .
```

## Running the Application

Because the project uses a `src` layout, set `PYTHONPATH` before starting the app.

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
uvicorn app.main:app --reload
```

On macOS/Linux:

```bash
export PYTHONPATH=src
uvicorn app.main:app --reload
```

The app will start at:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/api/v1/openapi.json`

## API Endpoints

### Root

```http
GET /
```

Returns a simple health/status payload:

```json
{
  "status": "online",
  "system": "Enterprise AI/ML FastAPI Platform"
}
```

### Classification Inference

```http
POST /api/v1/inference/predict/classification
```

Request body example:

```json
{
  "inputs": [
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
      "native-country": "United-States"
    }
  ]
}
```

Response example:

```json
{
  "prediction": "<=50K",
  "confidence": "86.25%"
}
```

### Regression Inference

```http
POST /api/v1/inference/predict/regression
```

Request body example:

```json
{
  "inputs": [
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
      "Condition": "Excellent"
    }
  ]
}
```

Response example:

```json
{
  "prediction": 245000.0,
  "confidence": 12000.5
}
```

## Model Loading Behavior

At startup, the app executes `ModelManager.load_all_models()`.

This loads every `.pkl` file in the `models/` directory and stores it in memory under the file name stem. For example:

- `income_salary_classification_model.pkl` -> `income_salary_classification_model`
- `house_price_regression_model.pkl` -> `house_price_regression_model`

If a model file is missing, the corresponding endpoint will return an HTTP 500 error with a message like:

```json
{
  "detail": "income_salary_classification_model not found."
}
```

## How the API Works

The inference service does the following:

1. Retrieves the required model from `ModelManager`
2. Converts the JSON payload into a pandas DataFrame
3. Calls the model's `predict()` method
4. Formats the result into a Pydantic response model
5. Adds confidence values when the model exposes probability or standard deviation data

## Notes

- CORS is enabled for all origins via `CORSMiddleware`
- The app is set up as a minimal ML serving layer, not a full business application
- Model compatibility depends on the structure and feature names expected by the saved `pkl` files

## Development Notes

If you want to extend this project:

- add new prediction routes under `src/app/modules/`
- keep request/response schemas in `schemas.py`
- keep model access logic inside the service layer
- add additional trained models to `models/` and they will be auto-discovered on startup

## License

This repository does not currently include a formal license file.