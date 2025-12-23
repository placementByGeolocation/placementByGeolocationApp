from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from app.repositories.history_repository import HistoryRepository

class HistoryService:
    def __init__(self, db: Session):
        self.repository = HistoryRepository(db)
    
    def save_request(
        self,
        endpoint: str,
        method: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ):
        """Сохранение запроса в историю"""
        return self.repository.create(
            endpoint=endpoint,
            method=method,
            input_data=input_data,
            output_data=output_data,
            headers=headers,
            status_code=status_code,
            error_message=error_message,
            processing_time_ms=processing_time_ms
        )
    
    def get_history_paginated(
        self,
        limit: int = 100,
        offset: int = 0,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None
    ) -> Tuple[List, int]:
        """Получение истории с пагинацией"""
        return self.repository.get_all(
            limit=limit,
            offset=offset,
            endpoint=endpoint,
            status_code=status_code
        )
    
    def get_record_by_id(self, record_id: int):
        """Получение записи по ID"""
        return self.repository.get_by_id(record_id)
    
    def cleanup_old_records(self, days_old: int = 30):
        """Очистка старых записей"""
        return self.repository.delete_old_records(days_old)