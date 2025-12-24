from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
import json
from app.repositories.history_repository import HistoryRepository

class HistoryService:
    def __init__(self, db: Session):
        self.repository = HistoryRepository(db)
    
    def save_request(
        self,
        endpoint: str,
        method: str,
        input_data: Any,
        output_data: Optional[Any] = None,
        headers: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[float] = None
    ):
        """Сохранение запроса в историю"""
        # Преобразуем данные в строки JSON
        if not isinstance(input_data, str):
            try:
                input_data = json.dumps(input_data, ensure_ascii=False)
            except (TypeError, ValueError):
                input_data = str(input_data)
        
        if output_data is not None and not isinstance(output_data, str):
            try:
                output_data = json.dumps(output_data, ensure_ascii=False)
            except (TypeError, ValueError):
                output_data = str(output_data)
        
        if headers is not None and not isinstance(headers, str):
            try:
                headers = json.dumps(headers, ensure_ascii=False)
            except (TypeError, ValueError):
                headers = str(headers)
        
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
