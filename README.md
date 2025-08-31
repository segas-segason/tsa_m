Работа с рядами в Python

Задание 1
<img width="741" height="689" alt="image" src="https://github.com/user-attachments/assets/d7a3a15c-eb07-4c05-9321-78d37e03d0a6" />


Ответ
```
# ============================================================
# Анализ временных рядов ВВП США
# gdp_t, y_t, Δy_t, Δ²y_t, дробные разности и scatter-графики
# ============================================================

# --- Импортируем библиотеки ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as web   # для загрузки данных FRED
from fracdiff import fracdiff               # для дробных разностей

plt.style.use("ggplot")  # красивый стиль графиков


# --- 1. Загружаем данные по ВВП ---
# GDP = Gross Domestic Product (Quarterly, SAAR)
gdp = web.DataReader('GDP', 'fred', start='1990-01-01')

# Индекс из FRED преобразуем в квартальный PeriodIndex
gdp.index = gdp.index.to_period('Q')

print("Первые наблюдения по ВВП:")
print(gdp.head())


# --- 2. Преобразования ряда ---
# Берём логарифм ВВП (часто используют для стабилизации дисперсии)
y = np.log(gdp['GDP'])

# Первая разность Δy_t = y_t - y_{t-1}
dy = y.diff()

# Вторая разность Δ²y_t = Δy_t - Δy_{t-1}
d2y = dy.diff()


# --- 3. Визуализация gdp_t, y_t, Δy_t, Δ²y_t ---
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

y.plot(ax=axes[0], title="Логарифм ВВП (y_t)")
dy.plot(ax=axes[1], title="Первая разность Δy_t")
d2y.plot(ax=axes[2], title="Вторая разность Δ²y_t")
gdp['GDP'].plot(ax=axes[3], title="Уровень ВВП (gdp_t)")

plt.tight_layout()
plt.show()


# --- 4. Дробные разности ---
# Подготовим данные (убираем NaN)
y_clean = y.dropna().values

# Δ^(1/2) y_t
y_fd05, _ = fracdiff(y_clean, d=0.5, window=5)

# Δ^(3/2) y_t
y_fd15, _ = fracdiff(y_clean, d=1.5, window=5)

# Преобразуем обратно в Series с тем же индексом
y_fd05 = pd.Series(y_fd05, index=y.dropna().index)
y_fd15 = pd.Series(y_fd15, index=y.dropna().index)

# Визуализация дробных разностей
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
y_fd05.plot(ax=axes[0], title="Дробная разность Δ^(1/2) y_t (окно=5)")
y_fd15.plot(ax=axes[1], title="Дробная разность Δ^(3/2) y_t (окно=5)")

plt.tight_layout()
plt.show()


# --- 5. Scatter-графики ---
# y_t vs y_{t-1}
sns.scatterplot(x=y.shift(1), y=y)
plt.title("y_t vs y_{t-1}")
plt.xlabel("y_{t-1}")
plt.ylabel("y_t")
plt.show()

# Δy_t vs Δy_{t-1}
sns.scatterplot(x=dy.shift(1), y=dy)
plt.title("Δy_t vs Δy_{t-1}")
plt.xlabel("Δy_{t-1}")
plt.ylabel("Δy_t")
```
plt.show()
