from fastapi import APIRouter
from app.api.endpoints import forward, history

router = APIRouter()

# Подключаем все эндпоинты
router.include_router(forward.router)
router.include_router(history.router)