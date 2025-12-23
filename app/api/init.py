from fastapi import APIRouter
from app.api.endpoints import forward, history

api_router = APIRouter()

# Подключаем все эндпоинты
api_router.include_router(forward.router)
api_router.include_router(history.router)