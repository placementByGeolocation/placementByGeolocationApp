from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Float
from sqlalchemy.sql import func
from app.core.database import Base
from datetime import datetime

class RequestHistory(Base):
    """Модель для хранения истории запросов"""
    __tablename__ = "request_history"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    input_data = Column(Text, nullable=False)  
    output_data = Column(Text, nullable=True)   
    headers = Column(Text, nullable=True)       
    status_code = Column(Integer, nullable=False)
    error_message = Column(Text, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<RequestHistory {self.id} - {self.endpoint}>"