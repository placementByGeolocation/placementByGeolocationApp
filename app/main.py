from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.core.config import settings
from app.core.database import Base, engine
from app.api import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events"""
    # Создаем таблицы
    Base.metadata.create_all(bind=engine)
    
    # Пробуем загрузить модель при старте
    try:
        from app.ml.model_loader import load_model
        model, feature_names = load_model()
        print(f"✅ ML Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"⚠️  ML Model not loaded: {e}")
    
    yield
    
    print("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )