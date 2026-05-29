import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

class GeolocationRequest(BaseModel):
    """
    Схема для входных данных с геолокацией и инфраструктурой.
    
    **Обязательные поля:**
    - lat, lon
    
    **Опциональные поля** (будут использованы значения по умолчанию, если не переданы):
    - city (дефолт: Москва)
    - infrastructure (дефолт: все 0)
    - nearest_distances (дефолт: все NaN)
    """
    lat: float = Field(..., description="Широта", example=55.7558)
    lon: float = Field(..., description="Долгота", example=37.6176)
    city: Optional[str] = Field(
        "Москва",
        description="Название города",
        example="Москва"
    )
    infrastructure: Optional[Dict[str, Dict[int, int]]] = Field(
        None,
        description="Кол-во объектов инфраструктуры в радиусах. Пример: {'metro': {500: 2, 1000: 5}, 'bus_stops': {500: 1, 1000: 3}}",
        example={"metro": {500: 2, 1000: 5}, "bus_stops": {500: 1, 1000: 3}}
    )
    nearest_distances: Optional[Dict[str, float]] = Field(
        None,
        description="Расстояния до ближайших объектов инфраструктуры в метрах. Пример: {'metro': 150.5, 'bus_stops': 200.0}",
        example={"metro": 150.5, "bus_stops": 200.0}
    )

class PredictionResponse(BaseModel):
    """Схема для ответа /forward"""
    success: bool
    prediction: Any
    timestamp: datetime
    processing_time_ms: Optional[float] = None
    input_metadata: Optional[Dict[str, Any]] = None
    
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

class HistoryResponse(BaseModel):
    """Схема для ответа /history"""
    id: int
    endpoint: str
    method: str
    status_code: int
    input_data: Union[Dict[str, Any], List[Any]]  # Может быть и словарем и списком
    output_data: Optional[Union[Dict[str, Any], List[Any]]]
    error_message: Optional[str]
    processing_time_ms: Optional[float]
    created_at: datetime
    
    @field_validator('input_data', 'output_data')
    def validate_data(cls, v):
        """Валидируем данные, преобразуя строки JSON в объекты"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HistoryListResponse(BaseModel):
    """Схема для списка истории"""
    count: int
    total_pages: int
    current_page: int
    results: List[HistoryResponse]

class StatsResponse(BaseModel):
    """Схема для ответа /stats"""
    success: bool
    timestamp: datetime
    filters_applied: Dict[str, Any]
    processing_time_stats: Optional[Dict[str, Any]] = None
    input_size_stats: Optional[Dict[str, Any]] = None
    input_field_stats: Optional[Dict[str, Any]] = None
    status_codes_distribution: Dict[int, int]
    parameter_distribution: Dict[str, Any]
    request_statistics: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }