from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.core.database import Base, engine
from app.core.ml_loader import ModelManager

# Import Module Routers
from app.modules.identity.router import router as identity_router
from app.modules.inference.router import router as inference_router
from fastapi.middleware.cors import CORSMiddleware

# Create DB Tables
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # App Startup: Pre-load ML Model
    ModelManager.load_all_models()
    yield
    # App Shutdown


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # সকল অরিজিন বা শুধু ["http://localhost:3000"] ব্যবহার করতে পারেন
    allow_credentials=True,
    allow_methods=["*"],  # সকল মেথড (POST, GET ইত্যাদি) এলাও করবে
    allow_headers=["*"],
)
# Register Module Routers
app.include_router(identity_router, prefix=settings.API_V1_STR)
app.include_router(inference_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "database_type": settings.DATABASE_TYPE,
    }
