import pickle
import os
from typing import Tuple, Any
from functools import lru_cache
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# Пути к файлам модели
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/feature_names.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/scaler.pkl")  # если нужен скейлер

class FixedMeanTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Mean Target Encoding только для строго фиксированных колонок.
    Никаких параметров cols снаружи нет.
    """

    FIXED_COLS = [
        'name', 'brand', 'nearest_museum_name',
        'nearest_theatre_name', 'nearest_gallery_name',
        'nearest_park_name'
    ]

    # если хотим - фиксируем сглаживание (можно 0.0)
    SMOOTHING = 0.0

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = np.asarray(y)

        self.global_mean_ = float(np.mean(y))
        self.mapping_ = {}

        cols_present = [c for c in self.FIXED_COLS if c in X.columns]

        for col in cols_present:
            tmp = pd.DataFrame({"x": X[col], "y": y})
            stats = tmp.groupby("x")["y"].agg(["mean", "count"])

            if self.SMOOTHING and self.SMOOTHING > 0:
                enc = (stats["count"] * stats["mean"] + self.SMOOTHING * self.global_mean_) / (stats["count"] + self.SMOOTHING)
            else:
                enc = stats["mean"]

            self.mapping_[col] = enc

        self.cols_ = cols_present
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in self.cols_:
            X[col] = X[col].map(self.mapping_[col]).fillna(self.global_mean_).astype(float)

        return X

# Регистрируем кастомные классы для pickle
CUSTOM_CLASSES = {
    'FixedMeanTargetEncoder': FixedMeanTargetEncoder,
    '__main__.FixedMeanTargetEncoder': FixedMeanTargetEncoder,
}

@lru_cache(maxsize=1)
def load_model() -> Tuple[Any, Any]:
    """Загружаем модель и имена фичей через pickle"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(FEATURE_NAMES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
            
        print(f"✅ Model loaded successfully. Features: {feature_names}")
        return model, feature_names
        
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print(f"Looking for files in: {MODEL_PATH}")
        raise
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
        self.model, self.feature_names = load_model()
        self.scaler = load_scaler()
        self.model_info = {
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "features": self.feature_names,
            "has_scaler": self.scaler is not None
        }
    
    def preprocess(self, features: list) -> np.ndarray:
        """Предобработка фичей"""
        # Проверяем количество фичей
        if self.feature_names and len(features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got {len(features)}"
            )
        
        features_array = np.array(features).reshape(1, -1)
        
        # Применяем скейлинг если есть скейлер
        if self.scaler:
            features_array = self.scaler.transform(features_array)
        
        return features_array
    
    def predict(self, features: list) -> Any:
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