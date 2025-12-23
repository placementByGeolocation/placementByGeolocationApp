from app.ml.model_loader import MLModel
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np

class MLService:
    def __init__(self):
        self.ml_model = MLModel()
    
    async def process_request(
        self, 
        features: List[float], 
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Обработка запроса для /forward"""
        try:
            # Предсказание модели
            prediction, probabilities = self.ml_model.predict(features)
            
            # Форматируем результат
            result = {
                "success": True,
                "prediction": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else str(prediction),
                "model_version": "1.0.0",
                "model_type": self.ml_model.model_info["model_type"],
                "timestamp": datetime.utcnow().isoformat(),
                "features_used": self.ml_model.model_info["features"],
                "features_count": len(features)
            }
            
            # Добавляем вероятности если есть
            if probabilities is not None:
                result["probabilities"] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
                result["confidence"] = float(np.max(probabilities))
            
            # Обрабатываем заголовки если нужно
            if "X-Model-Version" in headers:
                result["requested_version"] = headers["X-Model-Version"]
            
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
                'failed', 'error', 'invalid', 'validation'
            ]):
                raise Exception("Модель не смогла обработать данные")
            raise Exception(error_msg)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Информация о загруженной модели"""
        return self.ml_model.get_model_info()