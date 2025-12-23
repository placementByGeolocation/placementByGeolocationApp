from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import time
import json

from app.core.database import get_db
from app.services.ml_service import MLService
from app.services.history_service import HistoryService
from app.models.schemas import PredictionRequest, PredictionResponse, ErrorResponse
from datetime import datetime

router = APIRouter(prefix="/forward", tags=["ML Inference"])

@router.post(
    "/",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def forward_pass(
    request: PredictionRequest,
    request_obj: Request,
    db: Session = Depends(get_db),
    x_custom_header: Optional[str] = Header(None),
    x_model_version: Optional[str] = Header(None),
    x_features_count: Optional[int] = Header(None)
):
    """
    Эндпоинт для прогона данных через ML модель
    
    Headers:
    - X-Custom-Header: произвольный заголовок
    - X-Model-Version: версия модели (опционально)
    - X-Features-Count: ожидаемое количество фичей (для валидации)
    """
    start_time = time.time()
    history_service = HistoryService(db)
    
    try:
        # Получаем все заголовки
        headers = dict(request_obj.headers)
        
        # Дополнительная валидация через заголовки
        if x_features_count and len(request.features) != x_features_count:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {x_features_count} features, got {len(request.features)}"
            )
        
        # Инициализируем ML сервис
        ml_service = MLService()
        
        # Обработка через ML сервис
        result = await ml_service.process_request(
            features=request.features,
            headers=headers
        )
        
        # Вычисляем время обработки
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Сохраняем успешный запрос в историю
        history_record = history_service.save_request(
            endpoint="/forward",
            method="POST",
            input_data=request.model_dump(),
            output_data=result,
            headers=headers,
            status_code=200,
            processing_time_ms=processing_time_ms
        )
        
        # Добавляем ID запроса в ответ
        result["request_id"] = history_record.id
        result["processing_time_ms"] = processing_time_ms
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
        
    except Exception as e:
        # Определяем код ошибки
        error_detail = str(e)
        
        if "модель не смогла обработать данные" in error_detail:
            status_code = 403
            error_message = "Модель не смогла обработать данные"
        elif "validation" in error_detail.lower() or "expected" in error_detail.lower():
            status_code = 400
            error_message = f"bad request: {error_detail}"
        else:
            status_code = 400
            error_message = "bad request"
        
        # Вычисляем время обработки
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Сохраняем ошибку в историю
        history_service.save_request(
            endpoint="/forward",
            method="POST",
            input_data=request.model_dump(),
            output_data={"error": error_detail},
            headers=dict(request_obj.headers),
            status_code=status_code,
            error_message=error_detail,
            processing_time_ms=processing_time_ms
        )
        
        raise HTTPException(
            status_code=status_code,
            detail=error_message
        )