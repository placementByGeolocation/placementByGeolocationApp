from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
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
    """
    Получение истории всех запросов
    
    - **limit**: ограничение количества записей (макс. 1000)
    - **offset**: смещение для пагинации
    - **endpoint**: фильтрация по конкретному эндпоинту
    - **status_code**: фильтрация по коду ответа
    """
    history_service = HistoryService(db)
    
    # Получаем записи с фильтрацией
    records, total_count = history_service.get_history_paginated(
        limit=limit,
        offset=offset,
        endpoint=endpoint,
        status_code=status_code
    )
    
    # Вычисляем общее количество страниц
    total_pages = (total_count + limit - 1) // limit if limit > 0 else 1
    current_page = (offset // limit) + 1 if limit > 0 else 1
    
    # Преобразуем SQLAlchemy объекты в Pydantic модели
    history_results = [
        HistoryResponse.model_validate(record)
        for record in records
    ]
    
    return HistoryListResponse(
        count=len(records),
        total_pages=total_pages,
        current_page=current_page,
        results=history_results
    )

@router.get(
    "/{record_id}",
    response_model=HistoryResponse
)
async def get_history_record(
    record_id: int,
    db: Session = Depends(get_db)
):
    """
    Получение конкретной записи из истории по ID
    """
    history_service = HistoryService(db)
    
    record = history_service.get_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    return HistoryResponse.model_validate(record)