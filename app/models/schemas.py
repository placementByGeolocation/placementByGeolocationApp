from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class PredictionRequest(BaseModel):
    """Схема для входных данных /forward"""
    features: List[float] = Field(
        ...,
        description="Список признаков для модели",
        example=[5.1, 3.5, 1.4, 0.2],
        min_items=1
    )
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features list cannot be empty")
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("All features must be numbers")
        return v

class PredictionResponse(BaseModel):
    """Схема для ответа /forward"""
    success: bool
    prediction: Any
    model_version: str
    model_type: str
    features_used: List[str]
    features_count: int
    timestamp: datetime
    request_id: Optional[int] = None
    processing_time_ms: Optional[float] = None
    probabilities: Optional[List[float]] = None
    confidence: Optional[float] = None
    requested_version: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ErrorResponse(BaseModel):
    """Схема для ошибок"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ModelInfoResponse(BaseModel):
    """Схема для информации о модели"""
    model_type: str
    n_features: int
    features: List[str]
    has_scaler: bool
    version: str = "1.0.0"

class HistoryRecord(BaseModel):
    """Схема для записи истории"""
    id: int
    endpoint: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    created_at: datetime
    status_code: int