"""
Модуль для преобразования геолокации и других параметров в 313 признаков для модели
"""
import numpy as np
from typing import Dict, List, Optional, Any
import pandas as pd

class FeatureProcessor:
    """Преобразует входные данные в 313 признаков для модели"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.default_values = self._initialize_defaults()
        
    def _initialize_defaults(self) -> Dict[str, float]:
        """Инициализирует значения по умолчанию для всех признаков"""
        defaults = {}
        
        for feature in self.feature_names:
            # Устанавливаем значения по умолчанию в зависимости от типа признака
            if feature in ['lat', 'lon']:
                defaults[feature] = 0.0
            elif 'review_count' in feature.lower():
                defaults[feature] = 0.0
            elif 'competitors' in feature.lower():
                defaults[feature] = 0.0
            elif 'same_brand' in feature.lower():
                defaults[feature] = 0.0
            elif 'hhindex' in feature.lower():
                defaults[feature] = 0.5  # Индекс Херфиндаля-Хиршмана
            elif 'malls' in feature.lower() or 'stations' in feature.lower():
                defaults[feature] = 0.0
            elif 'nearest' in feature.lower():
                defaults[feature] = 1000.0  # Большое расстояние по умолчанию
            elif 'distance' in feature.lower():
                defaults[feature] = 5000.0  # Расстояние до центра города
            elif any(x in feature.lower() for x in ['facilities', 'schools', 'centres', 'museums', 'theatres', 'galleries', 'attractions', 'parks']):
                defaults[feature] = 0.0
            elif feature.startswith('type_'):
                defaults[feature] = 0.0  # Категориальные признаки типа заведения
            elif feature.startswith('cuisine_'):
                defaults[feature] = 0.0  # Категориальные признаки кухни
            else:
                defaults[feature] = 0.0
        
        return defaults
    
    def process_geolocation(self, lat: float, lon: float, **kwargs) -> List[float]:
        """
        Преобразует геолокацию и дополнительные параметры в 313 признаков
        
        Args:
            lat: Широта
            lon: Долгота
            **kwargs: Дополнительные параметры (тип заведения, кухня и т.д.)
        """
        # Начинаем со значений по умолчанию
        features = self.default_values.copy()
        
        # Обновляем координаты
        features['lat'] = float(lat)
        features['lon'] = float(lon)
        
        # Обрабатываем дополнительные параметры
        for key, value in kwargs.items():
            # Преобразуем ключ к формату признака
            feature_key = self._map_key_to_feature(key, value)
            if feature_key and feature_key in features:
                features[feature_key] = float(value) if isinstance(value, (int, float)) else 1.0
        
        # Преобразуем в список в правильном порядке
        result = []
        for feature_name in self.feature_names:
            result.append(features.get(feature_name, 0.0))
        
        return result
    
    def _map_key_to_feature(self, key: str, value: Any) -> Optional[str]:
        """Сопоставляет ключ из входных данных с именем признака"""
        key_lower = key.lower()
        
        # Проверяем тип заведения
        if key_lower in ['type', 'category', 'establishment_type']:
            if isinstance(value, str):
                return f"type_{value.lower().replace(' ', '_')}"
        
        # Проверяем кухню
        elif key_lower in ['cuisine', 'food_type', 'kitchen']:
            if isinstance(value, str):
                # Обрабатываем несколько кухонь через точку с запятой
                cuisines = value.split(';')
                if cuisines:
                    # Используем первую кухню или создаем комбинированную
                    if len(cuisines) == 1:
                        return f"cuisine_{cuisines[0].strip().lower().replace(' ', '_')}"
                    else:
                        # Ищем комбинированную кухню в признаках
                        combined = ';'.join([c.strip().lower().replace(' ', '_') for c in cuisines])
                        combined_feature = f"cuisine_{combined}"
                        if combined_feature in self.feature_names:
                            return combined_feature
        
        # Проверяем другие параметры
        elif key_lower in ['brand', 'name']:
            return key_lower
        elif 'competitor' in key_lower:
            return key_lower
        elif 'distance' in key_lower:
            return key_lower
        
        return None
    
    def validate_features(self, features: List[float]) -> bool:
        """Проверяет, что список признаков имеет правильную длину"""
        return len(features) == len(self.feature_names)

# Утилита для извлечения списка всех типов и кухонь из feature_names
def extract_categorical_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """Извлекает список всех категориальных признаков из имен признаков"""
    types = []
    cuisines = []
    
    for feature in feature_names:
        if feature.startswith('type_'):
            types.append(feature[5:])  # Убираем 'type_'
        elif feature.startswith('cuisine_'):
            cuisines.append(feature[8:])  # Убираем 'cuisine_'
    
    return {
        'types': types,
        'cuisines': cuisines
    }