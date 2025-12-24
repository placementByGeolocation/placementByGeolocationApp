from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.core.config import settings
from app.core.database import Base, engine
from app.api import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events для создания таблиц при старте"""
    # Создаем таблицы при запуске (в продакшене используйте миграции!)
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")
    yield
    # Cleanup при завершении
    print("👋 Shutting down...")

app = FastAPI(
    # title=settings.APP_NAME,
    # version=settings.MODEL_VERSION,
    # debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "ML Inference API",
        "version": settings.MODEL_VERSION,
        "docs": "/docs",
        "endpoints": {
            "forward": "POST /api/v1/forward",
            "history": "GET /api/v1/history"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-01-15T10:30:00Z"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )