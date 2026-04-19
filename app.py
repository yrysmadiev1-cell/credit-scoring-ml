from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# 1. Инициализируем приложение
app = FastAPI(title="Halyk Bank Credit Scoring API", version="1.0")

# 2. Загружаем сохраненную модель и скейлер при старте сервера
model = joblib.load('xgb_scoring_model.pkl')
scaler = joblib.load('scaler.pkl')

# Точные названия колонок, на которых обучалась модель
FEATURE_NAMES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 
    'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
]

# 3. Описываем, как должен выглядеть входящий JSON от клиента
class ClientData(BaseModel):
    utilization: float        # Процент использования кредиток (напр., 0.5)
    age: int                  # Возраст
    past_due_30_59: int       # Просрочки 30-59 дней
    debt_ratio: float         # Долг / Доход (напр., 0.3)
    monthly_income: float     # Ежемесячный доход
    open_credit_lines: int    # Количество открытых кредитов
    times_90_late: int        # Просрочки более 90 дней
    real_estate_loans: int    # Ипотеки
    past_due_60_89: int       # Просрочки 60-89 дней
    dependents: int           # Иждивенцы (дети)

# 4. Создаем эндпоинт (точку входа) для предсказаний
@app.post("/predict")
def predict_default(client: ClientData):
    # Превращаем JSON в датафрейм pandas (с правильными названиями колонок)
    input_data = pd.DataFrame([[
        client.utilization, client.age, client.past_due_30_59,
        client.debt_ratio, client.monthly_income, client.open_credit_lines,
        client.times_90_late, client.real_estate_loans,
        client.past_due_60_89, client.dependents
    ]], columns=FEATURE_NAMES)
    
    # Масштабируем данные так же, как делали это при обучении
    scaled_data = scaler.transform(input_data)
    
    # Предсказываем ВЕРОЯТНОСТЬ дефолта
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Бизнес-логика: если шанс дефолта > 30%, отказываем
    decision = "Reject" if probability > 0.30 else "Approve"
    
    return {
        "probability_of_default": round(float(probability), 4),
        "business_decision": decision
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)