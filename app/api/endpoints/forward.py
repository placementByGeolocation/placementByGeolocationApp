import json
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.orm import Session
from typing import Optional
import time
from datetime import datetime

from app.core.database import get_db
from app.services.ml_service import MLService
from app.services.history_service import HistoryService
from app.models.schemas import GeolocationRequest, PredictionResponse, ErrorResponse

router = APIRouter(prefix="/forward", tags=["ML"])

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
    request: GeolocationRequest,
    request_obj: Request,
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для прогона данных через ML модель (313 признаков)
    
    Формат входных данных - геолокация:
    ```json
    {
        "lat": 55.7558,
        "lon": 37.6176,
        "establishment_type": "restaurant",
        "cuisine": "italian",
        "brand": "Вкусно и точка",
        "additional_params": {
            "competitors_count": 5,
            "distance_to_metro": 300
        }
    }
    ```
    """
    start_time = time.time()
    history_service = HistoryService(db)
    
    try:
        # Получаем все заголовки
        headers = dict(request_obj.headers)
        
        # Инициализируем ML сервис
        ml_service = MLService()
        
        # Преобразуем геолокацию в 313 признаков
        features = ml_service.process_geolocation_request(
            lat=request.lat,
            lon=request.lon,
            establishment_type=request.establishment_type,
            cuisine=request.cuisine,
            brand=request.brand,
            **(request.additional_params or {})
        )
        
        input_metadata = {
            "type": "geolocation",
            "lat": request.lat,
            "lon": request.lon,
            "establishment_type": request.establishment_type,
            "cuisine": request.cuisine,
            "brand": request.brand,
            "additional_params": request.additional_params
        }
        
        result = await ml_service.process_request(features=features)
        
        # Вычисляем время обработки
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Добавляем недостающие поля в результат
        result.update({
            "input_format": "geolocation",
            "features_count": len(features),
            "model_type": ml_service.ml_model.model_info["model_type"],
            "input_metadata": input_metadata,
            "processing_time_ms": processing_time_ms
        })
        
        # Сохраняем успешный запрос в историю
        # При сохранении в историю, преобразуем данные в строку JSON, иначе плохо работало
        request_data = request.model_dump()
        input_data_to_save = json.dumps(request_data) if not isinstance(request_data, str) else request_data
        
        # Для response также
        output_data_to_save = json.dumps(result) if not isinstance(result, str) else result
        
        history_service.save_request(
            endpoint="/forward",
            method="POST",
            input_data=input_data_to_save,
            output_data=output_data_to_save,
            headers=headers,
            status_code=200,
            processing_time_ms=processing_time_ms
        )
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
        
    except Exception as e:
        # Определяем код ошибки
        error_detail = str(e)
        error_lower = error_detail.lower()
        
        if any(phrase in error_lower for phrase in [
            'модель не смогла', 'model failed', 'prediction failed',
            'validation error', 'data validation'
        ]):
            status_code = 403
            error_message = "Модель не смогла обработать данные"
        elif any(phrase in error_lower for phrase in [
            'expected', 'validation', 'invalid', 'bad request'
        ]):
            status_code = 400
            error_message = f"bad request: {error_detail}"
        else:
            status_code = 500
            error_message = "Internal server error"
        
        # Вычисляем время обработки
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Создаем объект ошибки для ответа
        error_response = ErrorResponse(
            success=False,
            error=error_message,
            detail=error_detail if error_detail != error_message else None,
            timestamp=datetime.now()
        )
        
        # Сохраняем ошибку в историю
        history_service.save_request(
            endpoint="/forward",
            method="POST",
            input_data=request.model_dump(),
            output_data=error_response.model_dump(),
            headers=dict(request_obj.headers),
            status_code=status_code,
            error_message=error_detail,
            processing_time_ms=processing_time_ms
        )
        
        raise HTTPException(
            status_code=status_code,
            detail=error_message
        )

# ЧТОБ БЫЛО, ВДРУГ ПОНАДОБИТСЯ
@router.get("/features-info")
async def get_features_info():
    """Получить информацию о требуемых признаках"""
    try:
        ml_service = MLService()
        return {
            "success": True,
            "total_features": len(ml_service.ml_model.feature_names),
            "feature_names": ml_service.ml_model.feature_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))