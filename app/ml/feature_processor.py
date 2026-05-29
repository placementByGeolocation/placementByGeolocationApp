"""
Модуль для преобразования геолокации и инфраструктуры в признаки для модели ATM placement.

Признаки строятся на основе:
- Координат (lat, lon)
- Количества объектов инфраструктуры в радиусах 500м и 1000м
- Расстояния до ближайшего объекта каждой категории
- Расстояния до центра города
- Города (категориальный признак)

Кошлки инфраструктуры: metro, bus_stops, malls, business_centres,
                       universities, schools, hospitals, parks
"""
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd


class FeatureProcessor:
    """
    Преобразует геолокацию и данные об инфраструктуре в признаки для модели CatBoost.
    
    Инициализируется с полным списком feature_names из обученной модели.
    """
    
    # Категории инфраструктуры и допустимые радиусы (в метрах)
    INFRASTRUCTURE_CATEGORIES = ['metro', 'bus_stops', 'malls', 'business_centres',
                                  'universities', 'schools', 'hospitals', 'parks']
    RADII = [500, 1000]
    
    # Центры городов (lat, lon)
    CITY_CENTERS = {
        "Москва": (55.7558, 37.6173),
        "Санкт-Петербург": (59.9343, 30.3351),
        "Нижний Новгород": (56.2965, 43.9361),
        "Новосибирск": (55.0084, 82.9357),
        "Казань": (55.7879, 49.1233),
    }
    
    def __init__(self, feature_names: List[str], infrastructure_data: Optional[Dict[str, List[Tuple[float, float]]]] = None):
        """
        Args:
            feature_names: Список имён признаков из модели (в правильном порядке)
            infrastructure_data: Опциональный словарь с данными об инфраструктуре.
                                Формат: {category: [(lat1, lon1), (lat2, lon2), ...]}
                                Если не provided, все count будут 0 и расстояния NaN.
        """
        self.feature_names = feature_names
        
        # Преобразуем данные инфраструктуры в numpy arrays для эффективного доступа
        self.infrastructure_points = {}
        if infrastructure_data is not None:
            for category in self.INFRASTRUCTURE_CATEGORIES:
                points = infrastructure_data.get(category, [])
                self.infrastructure_points[category] = np.array(points) if points else np.array([]).reshape(0, 2)
        else:
            for category in self.INFRASTRUCTURE_CATEGORIES:
                self.infrastructure_points[category] = np.array([]).reshape(0, 2)
    
    def process_geolocation(
        self,
        lat: float,
        lon: float,
        city: str,
    ) -> List[float]:
        """
        Преобразует геолокацию в вектор признаков, автоматически вычисляя
        признаки инфраструктуры на основе предоставленных данных.
        
        Args:
            lat: Широта
            lon: Долгота
            city: Название города
        
        Returns:
            Список значений признаков в порядке feature_names
        """
        # Инициализируем признаки со значениями по умолчанию
        features = self._get_default_features()
        
        # Устанавливаем координаты
        features['lat'] = float(lat)
        features['lon'] = float(lon)
        
        # Устанавливаем город (если он есть в feature_names)
        features['city'] = city
        
        # Вычисляем расстояние до центра города
        if 'distance_to_centre' in features:
            features['distance_to_centre'] = self._calculate_distance_to_centre(lat, lon, city)
        
        # Автоматически вычисляем признаки инфраструктуры
        for category in self.INFRASTRUCTURE_CATEGORIES:
            points = self.infrastructure_points[category]
            # Если есть точки для этой категории, вычисляем count и nearest distance
            if len(points) > 0:
                # Counts in 500m and 1000m radii
                count_500 = self._count_within_radius(lat, lon, points, 500)
                count_1000 = self._count_within_radius(lat, lon, points, 1000)
                features[f'{category}_500m'] = float(count_500)
                features[f'{category}_1000m'] = float(count_1000)
                # Nearest distance
                features[f'nearest_{category}'] = self._nearest_distance(lat, lon, points)
            # Если точек нет, оставляем значения по умолчанию (0 для count, NaN для nearest)
        
        # Преобразуем в вектор в правильном порядке
        result = []
        for feature_name in self.feature_names:
            result.append(features.get(feature_name, 0.0))
        
        return result
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Возвращает словарь признаков со значениями по умолчанию"""
        features = {}
        
        for feature_name in self.feature_names:
            if feature_name in ['lat', 'lon']:
                features[feature_name] = 0.0
            elif feature_name == 'city':
                features[feature_name] = 'Москва'  # Дефолтный город
            elif feature_name == 'distance_to_centre':
                features[feature_name] = 0.0
            elif feature_name.endswith('_500m') or feature_name.endswith('_1000m'):
                # Признаки количества инфраструктуры
                features[feature_name] = 0.0
            elif feature_name.startswith('nearest_'):
                # Признаки расстояния до ближайшей инфраструктуры
                features[feature_name] = np.nan
            else:
                features[feature_name] = 0.0
        
        return features
    
    def _calculate_distance_to_centre(self, lat: float, lon: float, city: str) -> float:
        """Вычисляет расстояние от точки до центра города в метрах"""
        if city not in self.CITY_CENTERS:
            return 0.0
        
        clat, clon = self.CITY_CENTERS[city]
        
        # Формула для расстояния в метрах (приблизительная, для малых расстояний)
        dlat = (lat - clat) * 111320
        dlon = (lon - clon) * 111320 * np.cos(np.radians(clat))
        distance = np.sqrt(dlat**2 + dlon**2)
        
        return float(distance)

    def _count_within_radius(self, lat: float, lon: float, points: np.ndarray, radius: float) -> int:
        """
        Counts the number of points within a given radius (in meters) from the given (lat, lon).
        Uses equirectangular approximation for distance.
        
        Args:
            lat: Latitude of the center point
            lon: Longitude of the center point
            points: Numpy array of shape (n, 2) where each row is [lat, lon] of a point
            radius: Radius in meters
        
        Returns:
            Number of points within the radius
        """
        if len(points) == 0:
            return 0
        # Vectorized distance calculation
        dlat = (lat - points[:, 0]) * 111320
        dlon = (lon - points[:, 1]) * 111320 * np.cos(np.radians(points[:, 0]))
        dists = np.sqrt(dlat**2 + dlon**2)
        return np.sum(dists <= radius).item()

    def _nearest_distance(self, lat: float, lon: float, points: np.ndarray) -> float:
        """
        Finds the minimum distance from the given (lat, lon) to any point in the points array.
        Returns NaN if the points array is empty.
        
        Args:
            lat: Latitude of the center point
            lon: Longitude of the center point
            points: Numpy array of shape (n, 2) where each row is [lat, lon] of a point
        
        Returns:
            Minimum distance in meters, or NaN if no points
        """
        if len(points) == 0:
            return np.nan
        # Vectorized distance calculation
        dlat = (lat - points[:, 0]) * 111320
        dlon = (lon - points[:, 1]) * 111320 * np.cos(np.radians(points[:, 0]))
        dists = np.sqrt(dlat**2 + dlon**2)
        return np.min(dists).item()
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет обработку ко всему датафрейму.
        
        Датафрейм должен содержать колонки:
        - lat, lon
        - city
        
        Args:
            df: Датафрейм с данными о ячейках сетки
        
        Returns:
            Датафрейм с колонками в порядке feature_names
        """
        # Убедимся, что есть координаты
        required_cols = ['lat', 'lon']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Мы будем вычислять признаки для каждой строки
        # Используем apply для простоты, но учтите, что это может быть медленно для больших датафреймов
        # и большого количества точек инфраструктуры.
        def process_row(row):
            lat = row['lat']
            lon = row['lon']
            city = row['city'] if city else 'Москва'  
            return self.process_geolocation(lat, lon, city)
        
        # Применяем функцию к каждой строке и создаем новый датафрейм
        features_list = df.apply(process_row, axis=1).tolist()
        result_df = pd.DataFrame(features_list, columns=self.feature_names)
        return result_df

# Утилита для извлечения списка инфраструктурных признаков из feature_names
def extract_infrastructure_features(feature_names: List[str]) -> Dict[str, List[str]]:
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
            # Извлекаем имя категории
            cat = feature.rsplit('_', 1)[0]
            categories.add(cat)
        elif feature.startswith('nearest_'):
            distances.append(feature)
            cat = feature[8:]  # Убираем 'nearest_'
            categories.add(cat)
    
    return {
        'counts': counts,
        'distances': distances,
        'categories': sorted(list(categories))
    }