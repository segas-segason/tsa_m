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


Задание 4

<img width="744" height="471" alt="image" src="https://github.com/user-attachments/assets/9ff44677-2539-4a6f-bdc2-2f416a7b30b8" />


Текст

```
\begin{exercise}
Рассмотрим месячные данные краткосрочной (3-х мес, \(rate1\)) и долгосрочной (10-ти лет., \(rate2\)))
ставкам для США с 1990-01-01 по н.в. как многомерный временной ряд \(rates\).
\begin{enumerate}
	\item Задайте временной индекс
	\item Визуализируйте ряд \(rates\) двумя способами
	\begin{itemize}
		\item раздельные графики
		\item общий график (два ряда на одном графике)
	\end{itemize}
	\item Визуализируйте ряд \(\diff rates\) двумя способами
	\item Визуализируйте ряд \(\diff^2 rates\) двумя способами
	\item Постройте гистограммы для \(rates,\diff rates,\diff^2 rates\) двумя способами
	\item Постройте диаграмму рассеяния \(rate1\) vs \(rate2\)
	\item Постройте диаграмму рассеяния \(\diff rate1 \) vs \(\diff rate2\)
\end{enumerate}
\end{exercise}
```

Ответ

```
# ============================================================
# Многомерный временной ряд ставок США: 3M (rate1), 10Y (rate2)
# Задачи: визуализация, разности, гистограммы, scatter
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

plt.style.use("ggplot")

# --- 1. Загрузка данных из FRED ---
start = "1990-01-01"
rate1 = web.DataReader("DGS3MO", "fred", start=start)  # 3M T-Bill
rate2 = web.DataReader("DGS10", "fred", start=start)   # 10Y Treasury

# --- 2. Приводим к месячной частоте (среднее по месяцу) ---
rate1_m = rate1.resample("M").mean()
rate2_m = rate2.resample("M").mean()

# --- 3. Объединяем в один DataFrame ---
rates = pd.concat([rate1_m, rate2_m], axis=1)
rates.columns = ["rate1", "rate2"]

print("Первые наблюдения:\n", rates.head())

# --- 4. Разности ---
diff1 = rates.diff().dropna()
diff2 = rates.diff().diff().dropna()

# ============================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================

# --- A. Раздельные графики (subplots) ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
rates["rate1"].plot(ax=axes[0], title="Краткосрочная ставка (3M)")
rates["rate2"].plot(ax=axes[1], title="Долгосрочная ставка (10Y)")
plt.tight_layout()
plt.show()

# --- B. Общий график ---
rates.plot(figsize=(10, 5), title="Ставки США (3M vs 10Y)")
plt.ylabel("Ставка, %")
plt.show()

# --- C. Разности (1-й порядок) ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
diff1["rate1"].plot(ax=axes[0], title="Δ rate1 (3M)")
diff1["rate2"].plot(ax=axes[1], title="Δ rate2 (10Y)")
plt.tight_layout()
plt.show()

diff1.plot(figsize=(10, 5), title="Δ rates (разности первого порядка)")
plt.show()

# --- D. Разности (2-й порядок) ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
diff2["rate1"].plot(ax=axes[0], title="Δ² rate1 (3M)")
diff2["rate2"].plot(ax=axes[1], title="Δ² rate2 (10Y)")
plt.tight_layout()
plt.show()

diff2.plot(figsize=(10, 5), title="Δ² rates (разности второго порядка)")
plt.show()

# --- E. Гистограммы (раздельные) ---
rates.hist(figsize=(10, 5), bins=30)
plt.suptitle("Гистограммы исходных ставок", y=1.02)
plt.show()

diff1.hist(figsize=(10, 5), bins=30)
plt.suptitle("Гистограммы Δ rates", y=1.02)
plt.show()

diff2.hist(figsize=(10, 5), bins=30)
plt.suptitle("Гистограммы Δ² rates", y=1.02)
plt.show()

# --- F. Гистограммы (на одном графике) ---
rates.plot(kind="hist", alpha=0.5, bins=30, figsize=(10, 5), title="Rates histogram (overlay)")
plt.show()

diff1.plot(kind="hist", alpha=0.5, bins=30, figsize=(10, 5), title="Δ rates histogram (overlay)")
plt.show()

diff2.plot(kind="hist", alpha=0.5, bins=30, figsize=(10, 5), title="Δ² rates histogram (overlay)")
plt.show()

# --- G. Диаграммы рассеяния ---
plt.figure(figsize=(6, 6))
plt.scatter(rates["rate1"], rates["rate2"], alpha=0.6)
plt.title("Scatter: rate1 vs rate2")
plt.xlabel("3M rate")
plt.ylabel("10Y rate")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(diff1["rate1"], diff1["rate2"], alpha=0.6, color="purple")
plt.title("Scatter: Δ rate1 vs Δ rate2")
plt.xlabel("Δ 3M rate")
plt.ylabel("Δ 10Y rate")
plt.grid(True)
plt.show()

```


Задание 5

<img width="777" height="346" alt="image" src="https://github.com/user-attachments/assets/88108831-ea63-499c-b33e-94eaebf444c4" />


Текст

```
\begin{exercise}
Рассмотрим месячные данные по США
\begin{itemize}
	\item краткосрочная (3-х мес) ставка
	\item долгосрочная (10-ти лет) ставка
	\item логарифм денежной массы M2
\end{itemize}
с 2000-01-01 по н.в. как многомерный временной ряд
\begin{enumerate}
	\item задайте временной индекс
	\item Визуализируйте многомерный ряд
	\item Визуализируйте первую и вторую разность
\end{enumerate}
\end{exercise}
```


Ответ

```
# --- файл: exercise_multivariate.py ---

# Импортируем библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Загружаем данные (допустим, у нас есть CSV с колонками: date, rate1, rate2, m2)
# Структура файла "us_data.csv":
# date,rate1,rate2,m2
# 2000-01-01,5.5,6.5,4500
# ...
data = pd.read_csv("us_data.csv", parse_dates=["date"])

# Устанавливаем временной индекс
data = data.set_index("date")

# Логарифмируем денежную массу M2 (новый столбец 'log_m2')
data["log_m2"] = np.log(data["m2"])

# Оставляем только нужные колонки: rate1, rate2, log_m2
df = data[["rate1", "rate2", "log_m2"]]

# --- 2. Визуализируем многомерный ряд ---
plt.style.use("ggplot")

# Способ 1: Раздельные графики
df.plot(subplots=True, figsize=(12, 8), title=["Rate1 (3m)", "Rate2 (10y)", "log(M2)"])
plt.suptitle("Многомерный временной ряд (раздельные графики)", y=1.02)
plt.show()

# Способ 2: Все на одном графике
df.plot(figsize=(12, 6), title="Многомерный временной ряд (общий график)")
plt.ylabel("Значение")
plt.show()

# --- 3. Визуализируем первую разность ---
df_diff1 = df.diff()

# Раздельные графики
df_diff1.plot(subplots=True, figsize=(12, 8), title=["ΔRate1", "ΔRate2", "Δlog(M2)"])
plt.suptitle("Первая разность", y=1.02)
plt.show()

# Общий график
df_diff1.plot(figsize=(12, 6), title="Первая разность (общий график)")
plt.ylabel("Δ значение")
plt.show()

# --- 4. Визуализируем вторую разность ---
df_diff2 = df.diff().diff()

# Раздельные графики
df_diff2.plot(subplots=True, figsize=(12, 8), title=["Δ²Rate1", "Δ²Rate2", "Δ²log(M2)"])
plt.suptitle("Вторая разность", y=1.02)
plt.show()

# Общий график
df_diff2.plot(figsize=(12, 6), title="Вторая разность (общий график)")
plt.ylabel("Δ² значение")
plt.show()

```
