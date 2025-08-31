Работа с рядами в Python

Задание 1

<img width="741" height="689" alt="image" src="https://github.com/user-attachments/assets/d7a3a15c-eb07-4c05-9321-78d37e03d0a6" />

Текст

```
\begin{exercise}
Рассмотрим \textbf{квартальные} данные по ВВП США с 1990 Q1 по н.в. (ряд \(gdp\))
и пусть \(y=\log(gdp)\)
\begin{enumerate}
	\item Задайте временной индекс
	\item Визуализируйте ряд \(gdp_t, y_t, \diff y_t, \diff^2 y_t\)
	\item Визуализируйте ряд \(\diff^{1/2} y_t, \diff^{3/2} y_t\) (ширина временного окна 5)
	\item Постройте диаграмму рассеяния \(y_t\) vs \(y_{t-1}\)
	\item Постройте диаграмму рассеяния \(\diff y_t\) vs \(\Delta y_{t-1}\)
	% \item вычислите \(\corr(y_t, y_{t-1})\) и 
	% тестируйте его значимость (формально!)
	% \item вычислите \(\corr(\diff y_t, \diff y_{t-1})\) и 
	% тестируйте его значимость (формально!) 
\end{enumerate}
\end{exercise}
```


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
plt.show()
```

Задание 2

<img width="746" height="266" alt="image" src="https://github.com/user-attachments/assets/dbeb147f-9556-43e5-b4a1-763da20b2d1a" />


Текст

```
\begin{exercise}
Рассмотрим \textbf{месячные} данные по M2 США с 1990-01-01 по н.в. (ряд \(m2\))
и пусть \(y=\log(m2)\)
\begin{enumerate}
	\item Задайте временной индекс
	\item Визуализируйте ряд \(m2, y_t, \diff y_t, \diff^2 y_t\)
	\item Визуализируйте ряд \(\diff^{1/2} y_t, \diff^{3/2} y_t\) (ширина временного окна 7)
	\item Постройте диаграмму рассеяния \(y_t\) vs \(y_{t-1}\)
	\item Постройте диаграмму рассеяния \(\diff y_t\) vs \(\Delta y_{t-1}\)
	% \item вычислите \(\corr(y_t, y_{t-1})\) и 
	% тестируйте его значимость (формально!)
	% \item вычислите \(\corr(\diff y_t, \diff y_{t-1})\) и 
	% тестируйте его значимость (формально!) 
\end{enumerate}
\end{exercise}
```


Ответ

```
# ============================================================
# Анализ временных рядов M2 США (месячные данные)
# m2_t, y_t, Δy_t, Δ²y_t, дробные разности и scatter-графики
# ============================================================

# --- Импорт библиотек ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as web   # загрузка из FRED
from fracdiff import fracdiff               # дробные разности

plt.style.use("ggplot")


# --- 1. Загружаем данные M2 ---
# M2SL = Money Stock, Monthly, Seasonally Adjusted
m2 = web.DataReader('M2SL', 'fred', start='1990-01-01')

# Индекс делаем месячным PeriodIndex
m2.index = m2.index.to_period('M')

print("Первые наблюдения M2:")
print(m2.head())


# --- 2. Преобразования ---
y = np.log(m2['M2SL'])    # логарифм
dy = y.diff()             # первая разность
d2y = dy.diff()           # вторая разность


# --- 3. Визуализация рядов ---
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

m2['M2SL'].plot(ax=axes[0], title="M2 (уровень)")
y.plot(ax=axes[1], title="Логарифм M2 (y_t)")
dy.plot(ax=axes[2], title="Первая разность Δy_t")
d2y.plot(ax=axes[3], title="Вторая разность Δ²y_t")

plt.tight_layout()
plt.show()


# --- 4. Дробные разности ---
# очищаем NaN
y_clean = y.dropna().values

# Δ^(1/2) y_t
y_fd05, _ = fracdiff(y_clean, d=0.5, window=7)

# Δ^(3/2) y_t
y_fd15, _ = fracdiff(y_clean, d=1.5, window=7)

# делаем Series с тем же индексом
y_fd05 = pd.Series(y_fd05, index=y.dropna().index)
y_fd15 = pd.Series(y_fd15, index=y.dropna().index)

# графики
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
y_fd05.plot(ax=axes[0], title="Дробная разность Δ^(1/2) y_t (окно=7)")
y_fd15.plot(ax=axes[1], title="Дробная разность Δ^(3/2) y_t (окно=7)")

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
plt.show()

```

Задание 3

<img width="725" height="153" alt="image" src="https://github.com/user-attachments/assets/f5f7f531-208c-4cc6-a2b8-ab25c95c8d7e" />

Текст

```
\begin{exercise}
Рассмотрим \textbf{недельные} данные по M2 США с 1990-01-01 по н.в.
\begin{enumerate}
	\item агрегируйте их в квартальные наблюдения (через усреднение)
	\item задайте временной индекс
	\item визуализируйте полученные наблюдения 
\end{enumerate}
\end{exercise}
```


Ответ

```
# ============================================================
# Недельные данные по M2 США (с 1990-01-01 по н.в.)
# Агрегация в квартальные наблюдения (средние значения)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

plt.style.use("ggplot")

# --- 1. Загрузка недельных данных M2 из FRED ---
# M2REALW или M2SL? -> Для недельных данных используем "M2REALW" (Weekly, Seasonally Adj.)
# Но FRED часто использует "M2SL" только в monthly. Для weekly правильный код: "M2REALW".
m2_weekly = web.DataReader('M2REALW', 'fred', start='1990-01-01')

print("Первые наблюдения (недельные):")
print(m2_weekly.head())

# --- 2. Агрегация в квартальные данные (средние значения) ---
m2_quarterly = m2_weekly.resample("Q").mean()

# --- 3. Приведение индекса к квартальному PeriodIndex ---
m2_quarterly.index = m2_quarterly.index.to_period("Q")

print("\nПервые квартальные наблюдения:")
print(m2_quarterly.head())

# --- 4. Визуализация ---
plt.figure(figsize=(10, 5))
plt.plot(m2_quarterly.index.to_timestamp(), m2_quarterly['M2REALW'], marker="o")
plt.title("Квартальные данные M2 (среднее за квартал)")
plt.ylabel("M2")
plt.xlabel("Год")
plt.grid(True)
plt.show()

```
