# Checkpoint 7 - наблюдаемость модели (MLflow + S3)

Логирование экспериментов и финальной модели размещения банкоматов в MLflow,
с хранением артефактов в S3 (MinIO), анализом ошибок и фиксацией PRD-версии.

## Состав
- `docker-compose.yml`, `Dockerfile.mlflow` - локальный MLflow + MinIO (S3) в Docker
- `requirements.txt` - зафиксированные версии для воспроизводимости
- `01_train_and_log.ipynb` - переобучение модели, логирование параметров/метрик,
  артефакты в S3, анализ ошибок, baseline, robustness, регистрация модели с тегом PRD
- `02_load_prd_and_predict.ipynb` - загрузка PRD-модели из реестра и тестовый предикт

## Модель
CatBoost на "умных негативах" (баланс 1:1: 10% случайных / 30% возле инфраструктуры /
60% соседних по координатам). Валидация - LeaveOneGroupOut по городам.
Метрики (LOGO-CV): ROC-AUC 0.8733, PR-AUC 0.8698, PR/baseline 1.76x. Seed = 42.

## Данные
Файл `grid_features.csv` (~354 МБ) в репозиторий не включён из-за размера.
Это сетка городских ячеек с инфраструктурными/POI-признаками (POI 500/1000м,
разнообразие POI, доля residential, число организаций и т.д.); таргет - наличие
банкомата в ячейке (`atm_count > 0`).
Скачать данные: https://disk.360.yandex.ru/d/A1o6ewSGJZZbWg
Положить `grid_features.csv` в корень рядом с блокнотами перед запуском.

## Как запустить
1. Поднять инфраструктуру: `docker compose up -d --build`
   (MLflow -> http://localhost:5000, MinIO -> http://localhost:9001, логин/пароль minioadmin).
   Серверу нужно ~1-2 минуты на прогрев.
2. Поставить зависимости: `pip install -r requirements.txt`.
3. Положить `grid_features.csv` в корень.
4. Прогнать `01_train_and_log.ipynb`, затем `02_load_prd_and_predict.ipynb`.
