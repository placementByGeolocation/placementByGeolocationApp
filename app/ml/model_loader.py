import pickle
import os
import sys
from typing import Tuple, List, Dict
from functools import lru_cache
import numpy as np
import pandas as pd

# Импортируем кастомные классы ДО загрузки модели
from app.ml.encoders import FixedMeanTargetEncoder

# Пути к файлам модели
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/feature_names.pkl")

# Регистрируем кастомные классы для pickle
CUSTOM_CLASSES = {
    'FixedMeanTargetEncoder': FixedMeanTargetEncoder,
    '__main__.FixedMeanTargetEncoder': FixedMeanTargetEncoder,
}

def custom_unpickler(file):
    """Кастомный анпиклер для обработки кастомных классов"""
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            full_name = f"{module}.{name}"
            
            if full_name in CUSTOM_CLASSES:
                return CUSTOM_CLASSES[full_name]
            
            if name in CUSTOM_CLASSES:
                return CUSTOM_CLASSES[name]
            
            if module == "__main__":
                if name in CUSTOM_CLASSES:
                    return CUSTOM_CLASSES[name]
                try:
                    return getattr(sys.modules['app.ml.encoders'], name)
                except:
                    pass
            
            return super().find_class(module, name)
    
    return CustomUnpickler(file)

@lru_cache(maxsize=1)
def load_model() -> Tuple[any, List[str]]:
    try:
        print(f"Loading model from: {MODEL_PATH}")
        
        # Загружаем модель
        with open(MODEL_PATH, 'rb') as f:
            model = custom_unpickler(f).load()
        
        # Загружаем feature_names
        feature_names = []
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                loaded = pickle.load(f)
                feature_names = loaded.tolist()
        
        print(f"✅ Model loaded. Features: {len(feature_names)}")
        return model, feature_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

class MLModel:
    def __init__(self):
        self.model, self.feature_names = load_model()
        # Получаем информацию о модели
        self.model_info = self._get_model_info()
    
    def _get_model_info(self) -> Dict:
        """Получаем информацию о модели"""
        info = {
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_names),
            "features": self.feature_names,
        }
        
        # Проверяем, pipeline ли это
        if hasattr(self.model, 'steps'):
            info["pipeline"] = True
            info["steps"] = [type(step[1]).__name__ for step in self.model.steps]
        else:
            info["pipeline"] = False
        
        return info
    
    def predict(self, features: List[float]) -> Tuple[any, any]:
        """Предсказание модели"""
        try:
            # Проверяем количество признаков
            if len(features) != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got {len(features)}"
                )
            
            # Создаем DataFrame для модели
            features_df = pd.DataFrame([features], columns=self.feature_names)
            
            # Предсказание
            prediction = self.model.predict(features_df)
            
            # Если модель возвращает вероятности
            # if hasattr(self.model, "predict_proba"):
            #     probabilities = self.model.predict_proba(features_df)
            #     return prediction[0], probabilities[0]
            # else:
            return prediction[0], None
                
        except Exception as e:
            raise Exception(f"Model prediction failed: {str(e)}")