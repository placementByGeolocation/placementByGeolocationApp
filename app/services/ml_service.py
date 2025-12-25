from app.ml.model_loader import MLModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

class MLService:
    def __init__(self):
        self.ml_model = MLModel()
    
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
    
    # TODO: фигня фигней, надо сделать нормальную обработку данных
    def process_geolocation_request(
        self,
        lat: float,
        lon: float,
        establishment_type: str = "restaurant",
        cuisine: str = "international",
        brand: Optional[str] = None,
        **kwargs
    ) -> List[float]:
        """
        Преобразует геолокацию и параметры в 313 признаков
        """
        # Создаем список из 313 нулей
        features = [0.0] * len(self.ml_model.feature_names)
        
        # Находим индексы нужных признаков
        feature_names = self.ml_model.feature_names
        
        # Устанавливаем координаты
        if 'lat' in feature_names:
            idx = feature_names.index('lat')
            features[idx] = float(lat)
        
        if 'lon' in feature_names:
            idx = feature_names.index('lon')
            features[idx] = float(lon)
        
        # Устанавливаем тип заведения
        type_feature = f"type_{establishment_type.lower().replace(' ', '_')}"
        if type_feature in feature_names:
            idx = feature_names.index(type_feature)
            features[idx] = 1.0
        
        # Устанавливаем кухню
        cuisine_feature = f"cuisine_{cuisine.lower().replace(' ', '_')}"
        if cuisine_feature in feature_names:
            idx = feature_names.index(cuisine_feature)
            features[idx] = 1.0
        
        # Устанавливаем бренд
        if brand and 'brand' in feature_names:
            idx = feature_names.index('brand')
            features[idx] = 1.0
        
        # Обрабатываем дополнительные параметры
        for key, value in kwargs.items():
            if key in feature_names:
                try:
                    idx = feature_names.index(key)
                    features[idx] = float(value)
                except (ValueError, TypeError):
                    features[idx] = 1.0 if value else 0.0
        
        return features
    