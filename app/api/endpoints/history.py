from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import json

from app.core.database import get_db
from app.models.domain import RequestHistory
from app.services.history_service import HistoryService
from app.models.schemas import HistoryListResponse, HistoryResponse

router = APIRouter(prefix="/history", tags=["History"])

@router.get(
    "/",
    response_model=HistoryListResponse
)
async def get_history(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=1000, description="Количество записей"),
    offset: int = Query(0, ge=0, description="Смещение"),
    endpoint: Optional[str] = Query(None, description="Фильтр по эндпоинту"),
    status_code: Optional[int] = Query(None, description="Фильтр по статусу")
):
    """Получение истории всех запросов"""
    history_service = HistoryService(db)
    
    # Получаем записи с фильтрацией
    records, total_count = history_service.get_history_paginated(
        limit=limit,
        offset=offset,
        endpoint=endpoint,
        status_code=status_code
    )
    
    total_pages = (total_count + limit - 1) // limit if limit > 0 else 1
    current_page = (offset // limit) + 1 if limit > 0 else 1
    
    # Преобразуем SQLAlchemy объекты с обработкой JSON
    history_results = []
    for record in records:
        # Парсим JSON строки в объекты
        input_data = record.input_data
        output_data = record.output_data

        if isinstance(input_data, str):
            input_data = json.loads(input_data)
        
        if isinstance(output_data, str):
            output_data = json.loads(output_data)
        
        history_result = {
            "id": record.id,
            "endpoint": record.endpoint,
            "method": record.method,
            "status_code": record.status_code,
            "input_data": input_data,
            "output_data": output_data,
            "error_message": record.error_message,
            "processing_time_ms": record.processing_time_ms,
            "created_at": record.created_at
        }
        
        history_results.append(HistoryResponse(**history_result))
    
    return HistoryListResponse(
        count=len(records),
        total_pages=total_pages,
        current_page=current_page,
        results=history_results
    )

# TODO: я это добавила потому что надо было почистить себе бд, надо добавить сюда какой-то там токен
# из задания, чтоб получить доп баллы
@router.delete("/", status_code=200)
async def clear_history(db: Session = Depends(get_db)):
    """Очистка всей истории запросов"""
    try:
        deleted_count = db.query(RequestHistory).delete()
        
        db.commit()
        return {
            "success": True,
            "message": f"Данные успешно очищены",
            "deleted_records": deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Ошибка очистки данных: {str(e)}")