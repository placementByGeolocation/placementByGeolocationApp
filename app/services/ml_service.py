from app.ml.model_loader import MLModel
from app.ml.feature_processor import FeatureProcessor
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

class MLService:
    def __init__(self):
        self.ml_model = MLModel()
        self.feature_processor = FeatureProcessor(self.ml_model.feature_names)
    
    async def process_request(
        self, 
        features: List[float]
    ) -> Dict[str, Any]:
        """Обработка запроса для /forward"""
        try:
            # Предсказание модели
            prediction, probabilities = self.ml_model.predict(features)
            
            # Форматируем результат
            result = {
                "success": True,
                "prediction": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else str(prediction),
                "timestamp": datetime.now().isoformat(),
            }
            
            # Добавляем вероятности если есть
            if probabilities is not None:
                result["probabilities"] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
                result["confidence"] = float(np.max(probabilities))
            
            return result
            
        except ValueError as e:
            # Ошибка валидации (неправильное количество фичей и т.д.)
            error_msg = f"Validation error: {str(e)}"
            raise Exception(error_msg)
            
        except Exception as e:
            # Ошибка модели
            error_msg = f"Model processing error: {str(e)}"
            
            # Определяем тип ошибки для правильного HTTP-кода
            if any(keyword in str(e).lower() for keyword in [
                'failed', 'error', 'invalid', 'validation', 'модель'
            ]):
                raise Exception("Модель не смогла обработать данные")
            raise Exception(error_msg)
    
    def process_geolocation_request(
        self,
        lat: float,
        lon: float,
        city: str,
    ) -> List[float]:
        """
        Преобразует геолокацию и данные об инфраструктуре в вектор признаков.
        
        Args:
            lat: Широта
            lon: Долгота
            city: Название города (Москва, Санкт-Петербург, и т.д.)
            infrastructure: Dict вида {category: {radius: count}}
                          Пример: {"metro": {500: 2, 1000: 5}, "bus_stops": {500: 1, 1000: 3}}
            nearest_distances: Dict вида {category: distance_in_meters}
                             Пример: {"metro": 150.5, "bus_stops": 200.0}
        
        Returns:
            Вектор признаков в правильном порядке для модели
        """
        return self.feature_processor.process_geolocation(
            lat=lat,
            lon=lon,
            city=city,
        )
    