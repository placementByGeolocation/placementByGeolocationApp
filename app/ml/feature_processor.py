"""
Модуль для поиска признаков в grid_features.csv по координатам.

Пользователь тыкает на карте -> получаем lat/lon -> ищем ближайший grid-квадрат
-> отбираем только нужные признаки -> подаем в модель.
"""
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from typing import List, Optional, Any
import os


class FeatureProcessor:
    """
    Находит ближайший grid-квадрат по координатам и возвращает признаки для модели.
    """
    
    # Центры городов (lat, lon)
    CITY_CENTERS = {
        "Москва": (55.7558, 37.6173),
        "Санкт-Петербург": (59.9343, 30.3351),
        "Нижний Новгород": (56.2965, 43.9361),
        "Новосибирск": (55.0084, 82.9357),
        "Казань": (55.7879, 49.1233),
    }
    
    def __init__(self, feature_names: List[str]):
        """
        Args:
            feature_names: Список имён признаков из модели (в правильном порядке)
        """
        self.feature_names = feature_names
        
        # Загружаем grid_features.csv
        self._load_grid_features()
        
    def _load_grid_features(self):
        """Загружает grid_features.csv из облака (Yandex Disk) и создает KD-дерево"""
        url = "https://disk.360.yandex.ru/d/A1o6ewSGJZZbWg"
        
        os.makedirs("app/models", exist_ok=True)
        local_path = "app/models/grid_features.csv"
        
        if not os.path.exists(local_path):
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        self.grid_df = pd.read_csv(local_path)
        
        coords = self.grid_df[['lat', 'lon']].values
        self.tree = cKDTree(coords)
        
    def _determine_city(self, lat: float, lon: float) -> str:
        """Определяет город по координатам, используя центры городов"""
        min_distance = float('inf')
        closest_city = "Москва"
        
        for city, (clat, clon) in self.CITY_CENTERS.items():
            dlat = (lat - clat) * 111320
            dlon = (lon - clon) * 111320 * np.cos(np.radians(clat))
            distance = np.sqrt(dlat**2 + dlon**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_city = city
                
        # Проверяем, что точка в пределах разумного расстояния от центра (например, 50km)
        if min_distance > 50000:
            return "Москва"  # дефолт
            
        return closest_city
    
    def process_geolocation(
        self,
        lat: float,
        lon: float,
        city: Optional[str] = None,
    ) -> List[float]:
        """
        Находит ближайший grid-квадрат и возвращает вектор признаков.
        
        Args:
            lat: Широта
            lon: Долгота
            city: Название города (опционально, определяется автоматически если не указан)
        
        Returns:
            Список значений признаков в порядке feature_names
        """
        # Ищем ближайший grid-квадрат
        _, idx = self.tree.query([lat, lon])
        row = self.grid_df.iloc[idx]
        
        # Определяем город если не указан
        if city is None or pd.isna(row.get('city')):
            city = self._determine_city(lat, lon)
        
        # Формируем результат - берем только нужные признаки
        result = []
        for feature_name in self.feature_names:
            if feature_name in self.grid_df.columns:
                val = row[feature_name]
                # Обрабатываем NaN
                if pd.isna(val):
                    result.append(0.0)
                elif feature_name == 'city':
                    result.append(city)
                else:
                    result.append(float(val))
            elif feature_name == 'city':
                result.append(city)
            else:
                result.append(0.0)
        
        return result
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет обработку ко всему датафрейму.
        
        Датафрейм должен содержать колонки lat, lon.
        
        Args:
            df: Датафрейм с координатами
        
        Returns:
            Датафрейм с колонками в порядке feature_names
        """
        required_cols = ['lat', 'lon']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Ищем ближайшие grid-квадраты для всех строк
        coords = df[['lat', 'lon']].values
        _, indices = self.tree.query(coords)
        
        # Берем строки из grid_df
        result_df = self.grid_df.iloc[indices].copy()
        
        # Обновляем city если нужно
        if 'city' not in df.columns:
            result_df['city'] = result_df.apply(
                lambda r: self._determine_city(r['lat'], r['lon']), axis=1
            )
        else:
            result_df['city'] = df['city'].values
        
        # Оставляем только нужные колонки
        available_features = [f for f in self.feature_names if f in result_df.columns]
        missing_features = [f for f in self.feature_names if f not in result_df.columns]
        
        if missing_features:
            for f in missing_features:
                result_df[f] = 0.0
        
        return result_df[self.feature_names]

# Утилита для извлечения списка инфраструктурных признаков из feature_names
def extract_infrastructure_features(feature_names: List[str]) -> dict:
    """
    Извлекает список всех инфраструктурных признаков из имен признаков.
    
    Returns:
        Словарь со следующей структурой:
        {
            'counts': [признаки вида '{category}_{radius}m'],
            'distances': [признаки вида 'nearest_{category}'],
            'categories': [список категорий инфраструктуры]
        }
    """
    counts = []
    distances = []
    categories = set()
    
    for feature in feature_names:
        if feature.endswith('m') and ('_500' in feature or '_1000' in feature):
            counts.append(feature)
            cat = feature.rsplit('_', 1)[0]
            categories.add(cat)
        elif feature.startswith('nearest_'):
            distances.append(feature)
            cat = feature[8:]
            categories.add(cat)
    
    return {
        'counts': counts,
        'distances': distances,
        'categories': sorted(list(categories))
    }