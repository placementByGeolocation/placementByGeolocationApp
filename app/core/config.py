from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    # App
    APP_NAME: str = "ML Service"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/ml_service.db"
    
    # ML
    ML_MODEL_PATH: str = f"{BASE_DIR}/app/ml/artifacts/model.pkl"
    MODEL_VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"

settings = Settings()