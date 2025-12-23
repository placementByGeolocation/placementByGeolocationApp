from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional, Tuple
from app.models.domain import RequestHistory

class HistoryRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create(self, **kwargs) -> RequestHistory:
        """Создание новой записи в истории"""
        record = RequestHistory(**kwargs)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record
    
    def get_all(
        self, 
        limit: int = 100,
        offset: int = 0,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None
    ) -> Tuple[List[RequestHistory], int]:
        """Получение записей с пагинацией и фильтрацией"""
        query = self.db.query(RequestHistory)
        
        # Применяем фильтры
        if endpoint:
            query = query.filter(RequestHistory.endpoint == endpoint)
        if status_code:
            query = query.filter(RequestHistory.status_code == status_code)
        
        # Получаем общее количество
        total_count = query.count()
        
        # Применяем сортировку и пагинацию
        records = query.order_by(desc(RequestHistory.created_at))\
                      .offset(offset)\
                      .limit(limit)\
                      .all()
        
        return records, total_count
    
    def get_by_id(self, record_id: int) -> Optional[RequestHistory]:
        """Получение записи по ID"""
        return self.db.query(RequestHistory)\
            .filter(RequestHistory.id == record_id)\
            .first()
    
    def delete_old_records(self, days_old: int = 30) -> int:
        """Удаление старых записей (для cleanup)"""
        from datetime import datetime, timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        deleted_count = self.db.query(RequestHistory)\
            .filter(RequestHistory.created_at < cutoff_date)\
            .delete()
        
        self.db.commit()
        return deleted_count