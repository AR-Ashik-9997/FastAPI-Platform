# FastAPI Platform

A lightweight FastAPI application that combines user identity management with an AI/ML inference layer. The project is organized into reusable modules and uses SQLAlchemy with SQLite by default, while supporting PostgreSQL and MySQL configuration via environment variables.

## Overview

This project is designed as an enterprise-style starter backend for:

- User/account CRUD features
- AI/ML model loading and inference endpoints
- Database-backed persistence through SQLAlchemy
- A modular FastAPI structure for future extension

The application loads an ML model at startup and exposes prediction endpoints for classification and regression tasks.

## Tech Stack

- Python 3.11+
- FastAPI
- SQLAlchemy
- Pydantic v2
- SQLite (default)
- PostgreSQL / MySQL support via configuration
- scikit-learn-compatible model files through joblib
- NumPy, Pandas

## Project Structure

```text
FastAPI-Platform/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── database.py
│   │   └── ml_loader.py
│   └── modules/
│       ├── identity/
│       │   ├── __init__.py
│       │   ├── models.py
│       │   ├── router.py
│       │   ├── schemas.py
│       │   └── service.py
│       └── inference/
│           ├── __init__.py
│           ├── router.py
│           ├── schemas.py
│           └── service.py
├── models/
│   └── trained_model.pkl
├── .env
├── requirements.txt
├── sql_app.db
└── README.md
```

## Configuration

The app reads configuration from `.env` using `pydantic-settings`.

Example configuration:

```env
DATABASE_TYPE=sqlite
SQLITE_DB_PATH=./sql_app.db

DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=secret
DB_NAME=ai_db
```

Supported database types:

- `sqlite`
- `postgresql`
- `mysql`

## Running the Application

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API server:

```bash
uvicorn app.main:app --reload
```

The service will be available at:

- `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/api/v1/openapi.json`

## API Endpoints

### Root

- `GET /` — health check endpoint

### Identity Module

- `GET /api/v1/identity/users` — list all users
- `GET /api/v1/identity/users/limit` — list users with pagination
- `GET /api/v1/identity/users/{user_id}` — fetch a single user
- `POST /api/v1/identity/users` — create a new user

Example payload for user creation:

```json
{
  "name": "John Doe",
  "email": "john@example.com"
}
```

### Inference Module

- `POST /api/v1/inference/predict/classification`
- `POST /api/v1/inference/predict/regression`

Example classification request:

```json
{
  "inputs": [
    {
      "age": 30,
      "workclass": "Private",
      "hours-per-week": 40
    }
  ]
}
```

## Database Model

The identity module contains a `User` table:

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
```

This model is created automatically at startup with `Base.metadata.create_all(bind=engine)`.

## ML Model Loading

The app tries to load a model file at startup from:

```text
models/trained_model.pkl
```

This is handled by `ModelManager.load_model()` in `app/core/ml_loader.py`.

If the model file is missing, the application still starts but the inference endpoints return an internal server error when invoked.

## Code Review Summary

This project has a clear modular FastAPI structure and a good separation of concerns:

- `app/core` handles configuration, database setup, and model loading
- `app/modules/identity` handles user management logic
- `app/modules/inference` handles ML prediction requests
- `app/main.py` wires the application together and registers routers

### Observed strengths

- Clean router/service separation
- SQLAlchemy models are straightforward
- Pydantic validation is used for API payloads
- Modular structure is easy to extend

### Observed issues and risks

- `HTTPException` is used in the identity router without being imported
- Some endpoints reference `HTTPException` but do not import it, which will raise runtime errors
- The inference request schema expects a list of feature dictionaries, but the code may be brittle when real ML models require different input structures
- The application assumes a trained model file exists and does not validate model compatibility before inference
- The default SQLite database file may be created in the repo root and should be managed carefully in production

## Suggested Next Improvements

- Fix missing imports and route-level exception handling
- Add proper validation for inference payloads based on the trained model
- Add unit tests for routes and service logic
- Add a `requirements.txt` lock workflow or Docker setup for consistent deployment
- Consider adding authentication and authorization if this becomes a production service
- Add a CI pipeline for linting and automated testing

## License

This project currently does not include a formal license file. If you plan to share or deploy it publicly, add a license definition such as MIT or Apache 2.0.
