from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
import json

from app.core.database import get_db
from app.services.ml_service import MLService
from app.services.history_service import HistoryService
from app.models.schemas import PredictionRequest, PredictionResponse, HistoryRecord

router = APIRouter()
ml_service = MLService()

@router.post("/forward", response_model=PredictionResponse)
async def forward_pass(
    request: PredictionRequest,
    request_obj: Request,
    custom_header: Optional[str] = Header(None, alias="X-Custom-Header"),
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для прогона данных через ML модель
    """
    history_service = HistoryService(db)
    
    try:
        # Получаем все заголовки
        headers = dict(request_obj.headers)
        
        # Обработка через ML сервис
        result = await ml_service.process_request(
            features=request.features,
            headers=headers
        )
        
        # Сохраняем в историю
        history_service.save_request(
            endpoint="/forward",
            input_data=request.dict(),
            output_data=result,
            status_code=200
        )
        
        return result
        
    except Exception as e:
        # Сохраняем ошибку
        history_service.save_request(
            endpoint="/forward",
            input_data=request.dict(),
            output_data={"error": str(e)},
            status_code=403,
            error_message=str(e)
        )
        raise HTTPException(status_code=403, detail=str(e))
        
    except Exception as e:
        history_service.save_request(
            endpoint="/forward",
            input_data=request.dict(),
            output_data={"error": "bad request"},
            status_code=400,
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail="bad request")

@router.get("/history")
async def get_history(
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Получение истории запросов"""
    history_service = HistoryService(db)
    records = history_service.get_history(limit)
    
    return {
        "count": len(records),
        "history": [
            {
                "id": record.id,
                "endpoint": record.endpoint,
                "input_data": record.input_data,
                "output_data": record.output_data,
                "status_code": record.status_code,
                "created_at": record.created_at.isoformat(),
                "error_message": record.error_message
            }
            for record in records
        ]
    }