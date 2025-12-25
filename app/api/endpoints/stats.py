from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.domain import RequestHistory

router = APIRouter(prefix="/stats", tags=["Statistics"])


def calculate_percentiles(data: List[float], percentiles: List[int]) -> Dict[int, float]:
    """Вычисляет перцентили для списка данных"""
    if not data:
        return {}
    
    sorted_data = sorted(data)
    result = {}
    
    for p in percentiles:
        if p < 0 or p > 100:
            continue
        
        index = (p / 100) * (len(sorted_data) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_data) - 1)
        
        if lower_index == upper_index:
            result[p] = sorted_data[lower_index]
        else:
            fraction = index - lower_index
            result[p] = sorted_data[lower_index] * (1 - fraction) + sorted_data[upper_index] * fraction
    
    return result


def parse_json_or_keep(value: Any) -> Any:
    """Пытается распарсить JSON, возвращает исходное значение в случае ошибки"""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@router.get("/")
async def get_stats(
    db: Session = Depends(get_db),
    hours: Optional[int] = None,
    endpoint: Optional[str] = None,
    status_code: Optional[int] = None
):
    """Получение статистики по запросам"""
    
    # Базовый запрос
    query = db.query(RequestHistory)
    
    # Применяем фильтры
    if hours:
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        query = query.filter(RequestHistory.created_at >= time_threshold)
    
    if endpoint:
        query = query.filter(RequestHistory.endpoint == endpoint)
    
    if status_code:
        query = query.filter(RequestHistory.status_code == status_code)
    
    # Получаем все записи
    records = query.order_by(desc(RequestHistory.created_at)).all()
    
    if not records:
        return {
            "success": False,
            "message": "No data available for the specified filters",
            "total_records": 0
        }
    
    # Анализируем данные
    processing_times = []
    input_sizes = []
    field_counts = []
    status_codes_dist = {}
    
    # Для анализа параметров
    establishment_types = {}
    cuisines = {}
    input_formats = {}
    
    for record in records:
        # Время обработки
        if record.processing_time_ms:
            processing_times.append(record.processing_time_ms)
        
        # Размер входных данных (в байтах)
        if record.input_data:
            input_size = len(str(record.input_data))
            input_sizes.append(input_size)
            
            # Парсим input_data для анализа
            parsed_input = parse_json_or_keep(record.input_data)
            
            # Определяем формат входных данных
            if isinstance(parsed_input, dict):
                input_formats["dict"] = input_formats.get("dict", 0) + 1
                field_counts.append(len(parsed_input))
                
                # Анализ параметров для геолокационных запросов
                if "lat" in parsed_input or "lon" in parsed_input:
                    # Геолокационный запрос
                    if isinstance(parsed_input, dict):
                        if "establishment_type" in parsed_input:
                            est_type = parsed_input.get("establishment_type")
                            if est_type:
                                establishment_types[str(est_type)] = establishment_types.get(str(est_type), 0) + 1
                        
                        if "cuisine" in parsed_input:
                            cuisine = parsed_input.get("cuisine")
                            if cuisine:
                                cuisines[str(cuisine)] = cuisines.get(str(cuisine), 0) + 1
            elif isinstance(parsed_input, list):
                input_formats["list"] = input_formats.get("list", 0) + 1
                field_counts.append(len(parsed_input))
            else:
                input_formats["other"] = input_formats.get("other", 0) + 1
                field_counts.append(1)  # Считаем как одно поле
        
        # Распределение статус-кодов
        status_codes_dist[record.status_code] = status_codes_dist.get(record.status_code, 0) + 1
    
    # Вычисляем статистику времени обработки
    time_stats = {}
    if processing_times:
        time_stats = {
            "count": len(processing_times),
            "mean": sum(processing_times) / len(processing_times),
            "min": min(processing_times),
            "max": max(processing_times),
            "percentiles": calculate_percentiles(processing_times, [50, 75, 90, 95, 99])
        }
    
    # Статистика по размеру входных данных (в байтах)
    size_stats = {}
    if input_sizes:
        size_stats = {
            "count": len(input_sizes),
            "mean_bytes": sum(input_sizes) / len(input_sizes),
            "min_bytes": min(input_sizes),
            "max_bytes": max(input_sizes),
            "percentiles_bytes": calculate_percentiles(input_sizes, [50, 75, 90, 95, 99])
        }
    
    # Статистика по количеству полей/элементов
    field_stats = {}
    if field_counts:
        field_stats = {
            "count": len(field_counts),
            "mean_fields": sum(field_counts) / len(field_counts),
            "min_fields": min(field_counts),
            "max_fields": max(field_counts),
            "percentiles_fields": calculate_percentiles(field_counts, [50, 75, 90, 95, 99])
        }
    
    # Сортируем распределения
    top_establishment_types = sorted(
        establishment_types.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    top_cuisines = sorted(
        cuisines.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    # Общая статистика
    total_stats = {
        "total_requests": len(records),
        "successful_requests": sum(1 for r in records if r.status_code == 200),
        "failed_requests": sum(1 for r in records if r.status_code != 200),
        "success_rate": (sum(1 for r in records if r.status_code == 200) / len(records)) * 100 if records else 0,
        "input_data_formats": input_formats
    }
    
    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "filters_applied": {
            "hours": hours,
            "endpoint": endpoint,
            "status_code": status_code
        },
        "processing_time_stats": time_stats,
        "input_size_stats": size_stats,
        "input_field_stats": field_stats,
        "status_codes_distribution": status_codes_dist,
        "parameter_distribution": {
            "top_establishment_types": dict(top_establishment_types),
            "top_cuisines": dict(top_cuisines)
        },
        "request_statistics": total_stats
    }
