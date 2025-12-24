# Регистрируем кастомные классы для pickle
from .encoders import FixedMeanTargetEncoder

# Экспортируем для импорта
__all__ = ['FixedMeanTargetEncoder']