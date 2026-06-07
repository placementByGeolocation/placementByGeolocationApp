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
            prediction, probabilities = self.ml_model.predict(features)
            
            result = {
                "success": True,
                "prediction": float(probabilities[1]) if isinstance(probabilities[1], (np.integer, np.floating)) else str(probabilities[1]),
                "timestamp": datetime.now().isoformat(),
            }
            
            if probabilities is not None:
                result["probabilities"] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
                result["confidence"] = float(np.max(probabilities))
            
            return result
            
        except ValueError as e:
            error_msg = f"Validation error: {str(e)}"
            raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"Model processing error: {str(e)}"
            
            if any(keyword in str(e).lower() for keyword in [
                'failed', 'error', 'invalid', 'validation', 'модель'
            ]):
                raise Exception("Модель не смогла обработать данные")
            raise Exception(error_msg)
    
    def process_geolocation_request(
        self,
        lat: float,
        lon: float,
        city: Optional[str] = None,
    ) -> List[float]:
        """
        Находит ближайший grid-квадрат по координатам и возвращает вектор признаков.
        
        Args:
            lat: Широта
            lon: Долгота
            city: Название города (опционально, определяется автоматически если не указан)
        
        Returns:
            Вектор признаков в правильном порядке для модели
        """
        return self.feature_processor.process_geolocation(
            lat=lat,
            lon=lon,
            city=city,
        )