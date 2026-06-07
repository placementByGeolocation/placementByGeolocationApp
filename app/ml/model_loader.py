import pickle
import os
import sys
from typing import Tuple, List, Dict, Any, Union
from functools import lru_cache
import numpy as np
import pandas as pd

# Импортируем кастомные классы ДО загрузки модели
from app.ml.encoders import FixedMeanTargetEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app/ml/artifacts/catboost_baseline.pkl")

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

        with open(MODEL_PATH, 'rb') as f:
            art = custom_unpickler(f).load()

        # Поддержка случая, когда в pickle сохранён не сама модель, а артефакт (dict)
        # с ключами 'model' и 'feature_cols' (см. README)
        model = art
        feature_names: List[str] = []
        if isinstance(art, dict):
            print("Detected artifact dict in pickle")
            # Предпочитаем точные ключи из README
            if 'model' in art:
                model = art['model']

            if 'feature_cols' in art and art['feature_cols'] is not None:
                feature_names = list(art['feature_cols'])
                print("Found feature names under key 'feature_cols'")
            else:
                # Фоллбек на другие возможные ключи
                for key in ('feature_names', 'feature_columns', 'features'):
                    if key in art and art[key] is not None:
                        feature_names = list(art[key])
                        print(f"Found feature names under key '{key}'")
                        break

        # ---------------------------------------------------
        # CASE 1: sklearn pipeline
        # ---------------------------------------------------

        if hasattr(model, 'steps'):

            print("Detected sklearn Pipeline")

            # последний step = модель
            final_estimator = model.steps[-1][1]

            print(f"Final estimator: {type(final_estimator).__name__}")

            # CatBoost
            if hasattr(final_estimator, 'feature_names_'):

                feature_names = list(final_estimator.feature_names_)

            elif hasattr(final_estimator, 'feature_names'):

                feature_names = list(final_estimator.feature_names)

        # ---------------------------------------------------
        # CASE 2: plain CatBoost
        # ---------------------------------------------------

        else:

            if hasattr(model, 'feature_names_'):

                feature_names = list(model.feature_names_)

            elif hasattr(model, 'feature_names'):

                feature_names = list(model.feature_names)

        # ---------------------------------------------------
        # FALLBACK
        # ---------------------------------------------------

        if not feature_names:
            print("WARNING: Feature names not found in model")

            # пробуем достать из sklearn metadata
            if hasattr(model, 'feature_names_in_'):

                feature_names = list(model.feature_names_in_)

        print(f"Model loaded successfully")
        print(f"Features count: {len(feature_names)}")
        print(f"Features: {feature_names}")

        return model, feature_names

    except Exception as e:

        print(f"Error loading model: {e}")
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

    def predict(self, features: Union[List[float], pd.DataFrame]) -> Tuple[Any, Any]:
        """Предсказание модели.

        Аргумент `features` может быть списком чисел (в порядке `feature_names`) или
        pandas.DataFrame с колонками, содержащими все `feature_names`.
        Возвращает кортеж `(prediction, probabilities)` — `probabilities` может быть `None`.
        """
        try:
            # Если передали DataFrame — используем его напрямую
            if isinstance(features, pd.DataFrame):
                features_df = features

                # Проверяем, что все нужные колонки присутствуют
                missing = [c for c in self.feature_names if c not in features_df.columns]
                if missing:
                    raise ValueError(f"Missing feature columns: {missing}")

                # Приводим порядок колонок и заполняем пропуски
                X = features_df[self.feature_names].fillna(0)

                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X)
                    preds = self.model.predict(X)
                    return preds[0] if len(preds) > 0 else preds, probs[0] if len(probs) > 0 else probs
                else:
                    preds = self.model.predict(X)
                    return preds[0], None

            # Иначе — ожидаем список признаков
            if isinstance(features, list) or isinstance(features, tuple) or isinstance(features, np.ndarray):
                if len(features) != len(self.feature_names):
                    raise ValueError(
                        f"Expected {len(self.feature_names)} features, got {len(features)}"
                    )

                features_df = pd.DataFrame([features], columns=self.feature_names).fillna(0)

                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(features_df)
                    preds = self.model.predict(features_df)
                    return preds[0], probs[0]
                else:
                    preds = self.model.predict(features_df)
                    return preds[0], None

            raise ValueError("Unsupported features type")

        except Exception as e:
            raise Exception(f"Model prediction failed: {str(e)}")