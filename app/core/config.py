from typing import Optional
from urllib.parse import quote_plus
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise AI/ML FastAPI Platform"
    API_V1_STR: str = "/api/v1"
    DATABASE_TYPE: str
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_NAME: Optional[str] = None
    SQLITE_DB_PATH: Optional[str] = None

    # Pydantic v2 Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field
    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        db_type = self.DATABASE_TYPE.lower()

        if db_type == "postgresql":
            password = quote_plus(self.DB_PASSWORD or "")
            return f"postgresql://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

        elif db_type == "mysql":
            password = quote_plus(self.DB_PASSWORD or "")
            return f"mysql+pymysql://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

        elif db_type == "sqlite":
            path = self.SQLITE_DB_PATH or "./sql_app.db"
            return f"sqlite:///{path}"

        else:
            raise ValueError(f"Unsupported DATABASE_TYPE: {self.DATABASE_TYPE}")


settings = Settings()