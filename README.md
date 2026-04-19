#  Credit Scoring Model & API

## О проекте
Полноценный микросервис для банковского кредитного скоринга. Модель предсказывает вероятность дефолта клиента (SeriousDlqin2yrs) на основе исторических финансовых данных и возвращает бизнес-решение (Approve / Reject) через REST API.

Проект демонстрирует полный цикл ML-разработки: от глубокого разведочного анализа данных (EDA) и обработки выбросов до подбора гиперпараметров и вывода модели в production с помощью FastAPI.

##  Технологический стек
* **Data Science & ML:** Python, Pandas, Scikit-Learn, XGBoost
* **MLOps / Backend:** FastAPI, Uvicorn, Joblib
* **Data Visualization:** Matplotlib, Seaborn

##  Ключевые этапы разработки
1. **Data Cleaning & EDA:** Обработка сильного дисбаланса классов (93% / 7%), импутация пропусков, винзоризация экстремальных выбросов (DebtRatio, возраст).
2. **Feature Engineering:** Масштабирование признаков (StandardScaler).
3. **Modeling:** * Построен базовый алгоритм `LogisticRegression` (class_weight='balanced').
   * Обучен алгоритм градиентного бустинга `XGBoost`.
   * Проведен подбор гиперпараметров через `GridSearchCV` с кросс-валидацией.
4. **Оценка модели:** Итоговый **ROC-AUC = 0.868**. Модель стабильна и не переобучена. Выявлены наиболее важные признаки (DebtRatio, Age, Utilization).

##  Как запустить проект локально

1. Склонируйте репозиторий:
```bash
git clone [https://github.com/твое-имя/credit-scoring-ml.git](https://github.com/твое-имя/credit-scoring-ml.git)
cd credit-scoring-ml