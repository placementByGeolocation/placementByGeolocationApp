import pickle
import os
import sys
from typing import Tuple, Any
from functools import lru_cache
import numpy as np

# Импортируем кастомные классы ДО загрузки модели
from app.ml.encoders import FixedMeanTargetEncoder

# Пути к файлам модели
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/feature_names.pkl")
# а нужен ли нам скалер вот так?
SCALER_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/scaler.pkl")

# Регистрируем кастомные классы для pickle
CUSTOM_CLASSES = {
    'FixedMeanTargetEncoder': FixedMeanTargetEncoder,
    '__main__.FixedMeanTargetEncoder': FixedMeanTargetEncoder,
}

# def custom_unpickler(file):
#     """Кастомный анпиклер для обработки кастомных классов"""
#     class CustomUnpickler(pickle.Unpickler):
#         def find_class(self, module, name):
#             # Ищем кастомные классы
#             full_name = f"{module}.{name}"
            
#             # Проверяем наш реестр кастомных классов
#             if full_name in CUSTOM_CLASSES:
#                 return CUSTOM_CLASSES[full_name]
            
#             # Проверяем по имени класса
#             if name in CUSTOM_CLASSES:
#                 return CUSTOM_CLASSES[name]
            
#             # Для классов из __main__
#             if module == "__main__":
#                 if name in CUSTOM_CLASSES:
#                     return CUSTOM_CLASSES[name]
#                 # Пробуем найти в нашем модуле
#                 try:
#                     return getattr(sys.modules['app.ml.encoders'], name)
#                 except:
#                     pass
            
#             # Стандартная загрузка
#             return super().find_class(module, name)
    
#     return CustomUnpickler(file)

@lru_cache(maxsize=1)
def load_model() -> Tuple[Any, Any]:
    """Загружаем модель и имена фичей через pickle с кастомным анпиклером"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        
        # Способ 1: Используем кастомный анпиклер
        # with open(MODEL_PATH, 'rb') as f:
        #     model = custom_unpickler(f).load()
        
        # TODO: разобраться какой метод использовать
        # Способ 2: Альтернативный метод (закомментировать способ 1 и раскомментировать этот)
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Загружаем feature_names
        feature_names = None
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
        
        print(f"✅ Model loaded successfully. Type: {type(model)}")
        if feature_names is not None:
            print(f"   Features: {feature_names}")
        
        return model, feature_names
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        raise
    # except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        print("Trying alternative loading method...")
        
        # Альтернативный метод загрузки
        try:
            # Пытаемся загрузить с явным импортом кастомных классов
            import sys
            sys.modules['__main__'].FixedMeanTargetEncoder = FixedMeanTargetEncoder
            
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            
            feature_names = None
            if os.path.exists(FEATURE_NAMES_PATH):
                with open(FEATURE_NAMES_PATH, 'rb') as f:
                    feature_names = pickle.load(f)
            
            print(f"✅ Model loaded (alternative method). Type: {type(model)}")
            return model, feature_names
            
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            raise Exception(f"Failed to load model: {e2}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

@lru_cache(maxsize=1)
def load_scaler() -> Any:
    """Загружаем скейлер если есть"""
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded successfully")
            return scaler
        except Exception as e:
            print(f"⚠️ Error loading scaler: {e}")
    return None

class MLModel:
    def __init__(self):
        print("Initializing MLModel...")
        self.model, self.feature_names = load_model()
        self.scaler = load_scaler()
        
        # Получаем информацию о модели
        self.model_info = self._get_model_info()
        print(f"Model info: {self.model_info}")
    
    def _get_model_info(self) -> dict:
        """Получаем информацию о модели"""
        info = {
            "model_type": type(self.model).__name__,
            "has_scaler": self.scaler is not None,
        }
        
        # Проверяем, есть ли у модели атрибуты
        if hasattr(self.model, 'feature_names_in_'):
            info["features"] = list(self.model.feature_names_in_)
            info["n_features"] = len(self.model.feature_names_in_)
        elif self.feature_names is not None:
            info["features"] = self.feature_names
            info["n_features"] = len(self.feature_names)
        else:
            info["features"] = None
            info["n_features"] = "unknown"
        
        # Проверяем, pipeline ли это
        if hasattr(self.model, 'steps'):
            info["pipeline"] = True
            info["steps"] = [type(step[1]).__name__ for step in self.model.steps]
        else:
            info["pipeline"] = False
        
        return info
    
    def preprocess(self, features: list) -> np.ndarray:
        """Предобработка фичей"""
        if self.model_info["n_features"] != "unknown":
            expected_features = self.model_info["n_features"]
            if len(features) != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {len(features)}"
                )
        
        features_array = np.array(features).reshape(1, -1)
        
        # Применяем скейлинг если есть скейлер
        if self.scaler:
            features_array = self.scaler.transform(features_array)
        
        return features_array
    
    def predict(self, features: list):
        """Предсказание модели"""
        try:
            # Предобработка
            processed_features = self.preprocess(features)
            
            # Предсказание
            prediction = self.model.predict(processed_features)
            
            # Если модель возвращает вероятности
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(processed_features)
                return prediction[0], probabilities[0]
            else:
                return prediction[0], None
                
        except Exception as e:
            raise Exception(f"Model prediction failed: {str(e)}")
    
    def get_model_info(self) -> dict:
        """Информация о модели"""
        return self.model_info