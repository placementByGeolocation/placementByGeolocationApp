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
    
    class Config:
        env_file = ".env"

settings = Settings()