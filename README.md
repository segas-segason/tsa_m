1 Работа с рядами в Python

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


Задание 6

<img width="725" height="249" alt="image" src="https://github.com/user-attachments/assets/e6f0d6d0-33aa-4eaf-9d2b-fd55d5f6f16f" />

Текст

```
\begin{exercise}
Из finance.yahoo.com загрузите данные по S\&P500 c 2005-01-01 по н.в. 
\begin{enumerate}
	\item Сформируйте месячный временной ряд из цены закрытия на последний день каждого месяца
	\item Задайте для него временной индекс
	\item Визуализируйте ряд
	\item Визуализируйте первую и вторую логарифмические разности
\end{enumerate}
\end{exercise}
```

Ответ

```
# --- exercise_sp500.py ---

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# 1. Загружаем данные по индексу S&P500 (^GSPC) с Yahoo Finance
sp500 = yf.download("^GSPC", start="2005-01-01")

# Берем только колонку 'Close'
sp500 = sp500[["Close"]]

# Формируем месячный ряд: берём последнюю цену в месяце (last)
sp500_monthly = sp500.resample("M").last()

# 2. Устанавливаем временной индекс (он уже есть после resample)
# Для удобства назовем колонку
sp500_monthly = sp500_monthly.rename(columns={"Close": "sp500"})

# --- 3. Визуализация исходного ряда ---
sp500_monthly.plot(figsize=(12, 6), title="S&P500: месячный ряд (цена закрытия)")
plt.ylabel("Индекс")
plt.show()

# --- 4. Логарифмические разности ---
log_sp500 = np.log(sp500_monthly)

# Первая разность (лог-доходности)
diff1 = log_sp500.diff()

# Вторая разность
diff2 = diff1.diff()

# Визуализация первой разности
diff1.plot(figsize=(12, 6), title="Первая логарифмическая разность (лог-доходности)")
plt.ylabel("Δ log(S&P500)")
plt.show()

# Визуализация второй разности
diff2.plot(figsize=(12, 6), title="Вторая логарифмическая разность")
plt.ylabel("Δ² log(S&P500)")
plt.show()

```


Задание 7

<img width="725" height="221" alt="image" src="https://github.com/user-attachments/assets/ef33515f-5c16-48ea-9a61-65ce5b4a1495" />


Текст

```
\begin{exercise}
Из finance.yahoo.com загрузите данные c 2005-01-01 по н.в. по
ценам закрытия S\&P500, Apple, Google
\begin{enumerate}
	\item Сформируйте многомерный ряд из цен закрытия на последний день каждого месяца
	\item Визуализируйте многомерный ряд
	\item Визуализируйте первую и вторую логарифмические разности
\end{enumerate}
\end{exercise}
```

Ответ

```
# --- exercise_sp500_aapl_goog.py ---

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# 1. Загружаем данные: S&P500 (^GSPC), Apple (AAPL), Google (GOOG)
tickers = ["^GSPC", "AAPL", "GOOG"]
data = yf.download(tickers, start="2005-01-01")["Close"]

# Формируем месячный ряд: берём последнюю цену в месяце
data_monthly = data.resample("M").last()

# Для удобства названия столбцов
data_monthly = data_monthly.rename(
    columns={"^GSPC": "S&P500", "AAPL": "Apple", "GOOG": "Google"}
)

# --- 2. Визуализация многомерного ряда ---
data_monthly.plot(figsize=(12, 6), title="Месячный ряд: цены закрытия")
plt.ylabel("Цена")
plt.show()

# --- 3. Первая и вторая логарифмические разности ---
log_data = np.log(data_monthly)

# Первая логарифмическая разность (лог-доходности)
diff1 = log_data.diff()

# Вторая логарифмическая разность
diff2 = diff1.diff()

# Визуализация первой разности
diff1.plot(figsize=(12, 6), title="Первая логарифмическая разность (лог-доходности)")
plt.ylabel("Δ log(price)")
plt.show()

# Визуализация второй разности
diff2.plot(figsize=(12, 6), title="Вторая логарифмическая разность")
plt.ylabel("Δ² log(price)")
plt.show()

```


2 ACF & PACF

Во всех задачах по умолчанию уровень значимости 5\%. !!!!

Задание 1

<img width="750" height="192" alt="image" src="https://github.com/user-attachments/assets/33c98560-5a29-45de-b741-6bde349019b7" />

Текст

```
\begin{exercise}
Рассмотрим \textbf{квартальные} данные по ВВП США с 1990 Q1 по н.в. (ряд \(gdp\))
и пусть \(y=\log(gdp)\)
\begin{enumerate}
	\item Постройте графики ACF и PACF для \(y_t, \diff y_t, \diff^2 y_t\)
	\item Значимы ли \(r(3),r_{part}(3)\) для \(\diff y_t\)?
	\item Вычислите  \(\{r(h)\}_{h=1}^3\) и \(\{r_{part}(h)\}_{h=1}^3\) для \(\diff y_t\)
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Задача 1: GDP США (квартальные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use("ggplot")

# 1. Загружаем данные по ВВП США с FRED
gdp = DataReader("GDP", "fred", "1990-01-01")  # квартальные данные
y = np.log(gdp['GDP'])  # логарифм ряда

# 2. Построение графиков ACF и PACF
def plot_acf_pacf(series, lags=20, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    fig.suptitle(title)
    plt.show()

plot_acf_pacf(y, title="GDP: log level")
plot_acf_pacf(y.diff(), title="GDP: first difference")
plot_acf_pacf(y.diff().diff(), title="GDP: second difference")

# 3. Вычисляем r(h) и r_part(h) для первых 3 лагов первой разности
diff_y = y.diff().dropna()

acf_vals = sm.tsa.stattools.acf(diff_y, nlags=3)
pacf_vals = sm.tsa.stattools.pacf(diff_y, nlags=3)

print("ACF r(h), h=1..3:", acf_vals[1:])
print("PACF r_part(h), h=1..3:", pacf_vals[1:])

# 4. Проверка значимости r(3) и r_part(3) на уровне 5%
n = len(diff_y)
threshold = 1.96 / np.sqrt(n)

r3_signif = abs(acf_vals[3]) > threshold
rpart3_signif = abs(pacf_vals[3]) > threshold

print(f"r(3) = {acf_vals[3]:.3f}, значим? {r3_signif}, порог ±{threshold:.3f}")
print(f"r_part(3) = {pacf_vals[3]:.3f}, значим? {rpart3_signif}, порог ±{threshold:.3f}")

```


Задание 2

<img width="732" height="197" alt="image" src="https://github.com/user-attachments/assets/332738f7-bc55-4534-9f9f-f91ffea2fc3d" />


Текст

```
\begin{exercise}
Рассмотрим \textbf{месячные} данные по M2 США с 1990-01-01 по н.в. (ряд \(m2\))
и пусть \(y=\log(m2)\)
\begin{enumerate}
	\item Постройте графики ACF и PACF для \(y_t, \diff y_t, \diff^2 y_t\)
	\item Значимы ли \(r(4),r_{part}(4)\) для \(\diff y_t\)?
	\item Вычислите  \(\{r(h)\}_{h=1}^3\) и \(\{r_{part}(h)\}_{h=1}^3\) для \(\diff y_t\)
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Задача 2: M2 США (месячные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use("ggplot")

# 1. Загружаем данные M2 с FRED
m2 = DataReader("M2SL", "fred", "1990-01-01")  # месячные данные
y = np.log(m2['M2SL'])  # логарифм ряда

# 2. Построение графиков ACF и PACF
def plot_acf_pacf(series, lags=20, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    fig.suptitle(title)
    plt.show()

plot_acf_pacf(y, title="M2: log level")
plot_acf_pacf(y.diff(), title="M2: first difference")
plot_acf_pacf(y.diff().diff(), title="M2: second difference")

# 3. Вычисляем r(h) и r_part(h) для первых 3 лагов первой разности
diff_y = y.diff().dropna()

acf_vals = sm.tsa.stattools.acf(diff_y, nlags=3)
pacf_vals = sm.tsa.stattools.pacf(diff_y, nlags=3)

print("ACF r(h), h=1..3:", acf_vals[1:])
print("PACF r_part(h), h=1..3:", pacf_vals[1:])

# 4. Проверка значимости r(4) и r_part(4) на уровне 5%
n = len(diff_y)
threshold = 1.96 / np.sqrt(n)

# Для 4-го лага
acf4 = sm.tsa.stattools.acf(diff_y, nlags=4)[4]
pacf4 = sm.tsa.stattools.pacf(diff_y, nlags=4)[4]

r4_signif = abs(acf4) > threshold
rpart4_signif = abs(pacf4) > threshold

print(f"r(4) = {acf4:.3f}, значим? {r4_signif}, порог ±{threshold:.3f}")
print(f"r_part(4) = {pacf4:.3f}, значим? {rpart4_signif}, порог ±{threshold:.3f}")

```


Задание 3

<img width="757" height="189" alt="image" src="https://github.com/user-attachments/assets/eef0cb45-6d8f-4808-bedb-bc7e282e5f7a" />


Текст

```
\begin{exercise}
Рассмотрим месячные данные по 3-х месячной ставки США с 2000-01 по н.в. (ряд \(y\))
\begin{enumerate}
	\item Постройте графики ACF и PACF для \(y_t, \diff y_t, \diff^2 y_t\)
	\item Значимы ли \(r(3),r_{part}(3)\) для \(\diff y_t\)?
	\item Вычислите  \(\{r(h)\}_{h=1}^3\) и \(\{r_{part}(h)\}_{h=1}^3\) для \(\diff y_t\)
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Задача 3: 3-месячная ставка США (месячные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use("ggplot")

# 1. Загружаем данные 3-месячной ставки США с FRED
# T-Bill 3-Month: "TB3MS"
y = DataReader("TB3MS", "fred", "2000-01-01")['TB3MS']

# 2. Построение графиков ACF и PACF
def plot_acf_pacf(series, lags=20, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    fig.suptitle(title)
    plt.show()

plot_acf_pacf(y, title="3M ставка: уровень")
plot_acf_pacf(y.diff(), title="3M ставка: первая разность")
plot_acf_pacf(y.diff().diff(), title="3M ставка: вторая разность")

# 3. Вычисляем r(h) и r_part(h) для первых 3 лагов первой разности
diff_y = y.diff().dropna()

acf_vals = sm.tsa.stattools.acf(diff_y, nlags=3)
pacf_vals = sm.tsa.stattools.pacf(diff_y, nlags=3)

print("ACF r(h), h=1..3:", acf_vals[1:])
print("PACF r_part(h), h=1..3:", pacf_vals[1:])

# 4. Проверка значимости r(3) и r_part(3) на уровне 5%
n = len(diff_y)
threshold = 1.96 / np.sqrt(n)

r3_signif = abs(acf_vals[3]) > threshold
rpart3_signif = abs(pacf_vals[3]) > threshold

print(f"r(3) = {acf_vals[3]:.3f}, значим? {r3_signif}, порог ±{threshold:.3f}")
print(f"r_part(3) = {pacf_vals[3]:.3f}, значим? {rpart3_signif}, порог ±{threshold:.3f}")

```


Задание 4

<img width="701" height="205" alt="image" src="https://github.com/user-attachments/assets/dd2c9499-3dea-450e-ada3-1a329df860c5" />


Текст

```
\begin{exercise}
Рассмотрим данные по S\&P500 с 2000-01 по н.в. (ряд \(sp500\))
и пусть \(y=\log(sp200)\)
\begin{enumerate}
	\item Постройте графики ACF и PACF для \(y_t, \diff y_t, \diff^2 y_t\)
	\item Значимы ли \(r(5),r_{part}(5)\) для \(\diff y_t\)?
	\item Вычислите  \(\{r(h)\}_{h=1}^3\) и \(\{r_{part}(h)\}_{h=1}^3\) для \(\diff y_t\)
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Задача 4: S&P500 (с 2000-01 по н.в.)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.style.use("ggplot")

# 1. Загружаем данные S&P500 через yfinance
sp500 = yf.download("^GSPC", start="2000-01-01")['Adj Close']

# 2. Создаем логарифм ряда
y = np.log(sp500)

# 3. Построение графиков ACF и PACF
def plot_acf_pacf(series, lags=20, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    fig.suptitle(title)
    plt.show()

plot_acf_pacf(y, title="S&P500: log level")
plot_acf_pacf(y.diff(), title="S&P500: first difference")
plot_acf_pacf(y.diff().diff(), title="S&P500: second difference")

# 4. Вычисляем r(h) и r_part(h) для первых 3 лагов первой разности
diff_y = y.diff().dropna()

acf_vals = sm.tsa.stattools.acf(diff_y, nlags=5)   # до 5 лагов для проверки
pacf_vals = sm.tsa.stattools.pacf(diff_y, nlags=5)

print("ACF r(h), h=1..3:", acf_vals[1:4])
print("PACF r_part(h), h=1..3:", pacf_vals[1:4])

# 5. Проверка значимости r(5) и r_part(5) на уровне 5%
n = len(diff_y)
threshold = 1.96 / np.sqrt(n)

r5_signif = abs(acf_vals[5]) > threshold
rpart5_signif = abs(pacf_vals[5]) > threshold

print(f"r(5) = {acf_vals[5]:.3f}, значим? {r5_signif}, порог ±{threshold:.3f}")
print(f"r_part(5) = {pacf_vals[5]:.3f}, значим? {rpart5_signif}, порог ±{threshold:.3f}")

```


3 Модель ARIMA 

Задание 1


<img width="784" height="339" alt="image" src="https://github.com/user-attachments/assets/57c26da2-e236-46d7-8a7c-086abd3adffb" />

Текст

```
\begin{exercise}
Рассмотрим модель ARIMA
\begin{align*}
	\phi(L)(1-L)^dy_t&=\theta(L)u_t & u_t&\sim WN(0,\sigma^2)
\end{align*}
для следующих многочленов
\begin{center}
	\begin{tabular}{l|c|c|c}
		\textnumero & \(d\) & \(\phi(z)\) & \(\theta(z)\) \\ \hline
		1 & 0 & \(1-z+0.25z^2\) & \(1+0.8z\) \\ \hline
		2 & 0 & \(1+0.8z-0.7z^2\) & \(1+0.5z-0.8z^2\) \\ \hline
		3 & 1 & \(1-0.2z+0.08z^2\) & \(1-0.3z-0.88z^2\) \\ \hline
		4 & 2 & \(1+1.9z-0.2z^2\) & \(1-1.6z-0.36z^2\) \\ \hline
	\end{tabular}
\end{center}
Для каждого случая проверьте условия стационарности и обратимости.
Запишите спецификацию модели без использования лагового оператор.
\end{exercise}
```

Ответ

```
# ======================================================
# ARIMA: проверка стационарности и обратимости
# ======================================================

import numpy as np
import pandas as pd

# Функция проверки корней и условий
def check_stationarity_invertibility(phi_coeffs, theta_coeffs, d=0):
    """
    phi_coeffs: коэффициенты AR-полинома [1, -phi1, -phi2, ...] (как в лаговом представлении)
    theta_coeffs: коэффициенты MA-полинома [1, theta1, theta2, ...]
    d: порядок интегрирования
    """
    # AR корни
    phi_roots = np.roots(phi_coeffs)
    ar_stationary = np.all(np.abs(phi_roots) > 1)
    
    # MA корни
    theta_roots = np.roots(theta_coeffs)
    ma_invertible = np.all(np.abs(theta_roots) > 1)
    
    return phi_roots, ar_stationary, theta_roots, ma_invertible

# ======================================================
# Случай 1
# phi(z) = 1 - z + 0.25 z^2
# theta(z) = 1 + 0.8 z
phi1 = [0.25, -1, 1]     # перевернули для numpy.roots
theta1 = [0.8, 1]
phi_roots, ar_stat, theta_roots, ma_inv = check_stationarity_invertibility(phi1, theta1)
print("Случай 1:")
print("AR корни:", phi_roots, "Стационарна?", ar_stat)
print("MA корни:", theta_roots, "Обратима?", ma_inv)
print("Модель без лагов: y_t - y_{t-1} +0.25 y_{t-2} = u_t + 0.8 u_{t-1}\n")

# ======================================================
# Случай 2
# phi(z) = 1 + 0.8 z -0.7 z^2
# theta(z) = 1 +0.5 z -0.8 z^2
phi2 = [-0.7, 0.8, 1]
theta2 = [-0.8, 0.5, 1]
phi_roots, ar_stat, theta_roots, ma_inv = check_stationarity_invertibility(phi2, theta2)
print("Случай 2:")
print("AR корни:", phi_roots, "Стационарна?", ar_stat)
print("MA корни:", theta_roots, "Обратима?", ma_inv)
print("Модель без лагов: y_t +0.8 y_{t-1} -0.7 y_{t-2} = u_t +0.5 u_{t-1} -0.8 u_{t-2}\n")

# ======================================================
# Случай 3
# phi(z) = 1 -0.2 z +0.08 z^2
# theta(z) = 1 -0.3 z -0.88 z^2
d3 = 1
phi3 = [0.08, -0.2, 1]
theta3 = [-0.88, -0.3, 1]
phi_roots, ar_stat, theta_roots, ma_inv = check_stationarity_invertibility(phi3, theta3, d=d3)
print("Случай 3 (d=1):")
print("AR корни:", phi_roots, "Стационарна после разности?", ar_stat)
print("MA корни:", theta_roots, "Обратима?", ma_inv)
print("Модель без лагов: (y_t - y_{t-1}) -0.2(y_{t-1}-y_{t-2}) +0.08(y_{t-2}-y_{t-3}) = u_t -0.3 u_{t-1} -0.88 u_{t-2}\n")

# ======================================================
# Случай 4
# phi(z) = 1 +1.9 z -0.2 z^2
# theta(z) = 1 -1.6 z -0.36 z^2
d4 = 2
phi4 = [-0.2, 1.9, 1]
theta4 = [-0.36, -1.6, 1]
phi_roots, ar_stat, theta_roots, ma_inv = check_stationarity_invertibility(phi4, theta4, d=d4)
print("Случай 4 (d=2):")
print("AR корни:", phi_roots, "Стационарна после второй разности?", ar_stat)
print("MA корни:", theta_roots, "Обратима?", ma_inv)
print("Модель без лагов: (y_t -2y_{t-1}+y_{t-2}) +1.9(y_{t-1}-2y_{t-2}+y_{t-3}) -0.2(y_{t-2}-2y_{t-3}+y_{t-4}) = u_t -1.6 u_{t-1}-0.36 u_{t-2}\n")

```

ARIMA в Python

Задание 1


<img width="707" height="1010" alt="image" src="https://github.com/user-attachments/assets/8b2d8afc-421b-40b3-a311-7992667d69e3" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- логарифм US GDP (\textbf{квартальные данные}) с 1995 по н.в.
\begin{enumerate}
	\item Подгонка модели заданного порядка
	\begin{enumerate}
		\item Подгоните модели
		\begin{center}\small
		\begin{tabular}{l|c|c}
			Модель & drift/trend  & спецификация\\ \hline
			ARIMA(1,0,1) & + & \(y_t=\alpha_0+\alpha_1t+\phi y_{t-1}+u_t+\theta u_{t-1}\)\\
			ARIMA(1,1,0) & + & \(\diff y_t=\alpha_0+\phi\diff y_{t-1}+u_t\)\\
			ARIMA(1,1,1) & - & \(\diff y_t=\phi\diff y_{t-1}+u_t+\theta u_{t-1}\) \\
			ARIMA(1,2,0) & - & \(\diff^2 y_t=\phi\diff^2 y_{t-1}+u_t\)\\ \hline
		\end{tabular}
		\end{center} 
		и постройте прогноз на 10 периодов. Значим ли снос/тренд?
		\item Проведите диагностику каждой модели.
		\item Проведите кросс-валидацию каждой модели. Какая предпочтительней?
	\end{enumerate}
	\item Примените тесты единичного корня и найдите порядок интегрирования для \(y_t\). 
	\item Подгонка <<оптимальной модели>>
	\begin{enumerate}
		\item Подгоните <<оптимальную>> модель ARIMA
		\item проведите её диагностику
		\item Постройте прогноз на 10 периодов
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Анализ US GDP: ARIMA подгонка, прогноз, тесты единичного корня
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf  # если нужно скачать данные из Yahoo

plt.style.use('ggplot')

# -----------------------------
# 1. Импорт данных
# -----------------------------
# Пример: если есть CSV с квартальным GDP
# gdp = pd.read_csv('US_GDP_quarterly.csv', parse_dates=['Date'], index_col='Date')

# Или через FRED
import pandas_datareader.data as web
gdp = web.DataReader('GDP', data_source='fred', start='1995-01-01', end='2023-12-31')

# Логарифм ряда
y = np.log(gdp['GDP'])

# Задаём временной индекс как квартальный
y.index = y.index.to_period('Q')
y.plot(title='Log of US GDP')
plt.show()

# -----------------------------
# 2. Подгонка моделей ARIMA
# -----------------------------

# ARIMA(1,0,1) с drift и trend
model_101 = ARIMA(y, order=(1,0,1), trend='ct')  # 'c' = drift, 't' = trend
res_101 = model_101.fit()
print(res_101.summary())

# Прогноз на 10 периодов
forecast_101 = res_101.get_forecast(steps=10)
forecast_101.predicted_mean.plot(title='Forecast ARIMA(1,0,1) with drift/trend')
plt.show()

# ARIMA(1,1,0) с drift
model_110 = ARIMA(y, order=(1,1,0), trend='c')
res_110 = model_110.fit()
print(res_110.summary())

forecast_110 = res_110.get_forecast(steps=10)
forecast_110.predicted_mean.plot(title='Forecast ARIMA(1,1,0) with drift')
plt.show()

# ARIMA(1,1,1) без drift/trend
model_111 = ARIMA(y, order=(1,1,1), trend='n')
res_111 = model_111.fit()
print(res_111.summary())

forecast_111 = res_111.get_forecast(steps=10)
forecast_111.predicted_mean.plot(title='Forecast ARIMA(1,1,1) no drift/trend')
plt.show()

# ARIMA(1,2,0) без drift/trend
model_120 = ARIMA(y, order=(1,2,0), trend='n')
res_120 = model_120.fit()
print(res_120.summary())

forecast_120 = res_120.get_forecast(steps=10)
forecast_120.predicted_mean.plot(title='Forecast ARIMA(1,2,0) no drift/trend')
plt.show()

# -----------------------------
# 3. Диагностика моделей
# -----------------------------
# Проверяем остатки: автокорреляция, нормальность
for res, name in zip([res_101, res_110, res_111, res_120],
                     ['ARIMA(1,0,1)', 'ARIMA(1,1,0)', 'ARIMA(1,1,1)', 'ARIMA(1,2,0)']):
    print(f"=== Диагностика {name} ===")
    res.plot_diagnostics(figsize=(10,8))
    plt.show()

# -----------------------------
# 4. Кросс-валидация (пример)
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
models = {'ARIMA101': (1,0,1), 'ARIMA110': (1,1,0), 'ARIMA111': (1,1,1), 'ARIMA120': (1,2,0)}

for name, order in models.items():
    errors = []
    for train_index, test_index in tscv.split(y):
        train, test = y.iloc[train_index], y.iloc[test_index]
        model = ARIMA(train, order=order, trend='c' if order[1]==0 or order[1]==1 else 'n').fit()
        pred = model.forecast(len(test))
        errors.append(np.mean((pred - test)**2))
    print(f'{name} CV MSE: {np.mean(errors)}')

# -----------------------------
# 5. Тест единичного корня (ADF)
# -----------------------------
adf_result = adfuller(y)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
if adf_result[1] < 0.05:
    print("Ряд стационарен")
else:
    print("Ряд нестационарен, нужно дифференцировать")

# -----------------------------
# 6. Автоматическая оптимальная ARIMA
# -----------------------------
auto_model = auto_arima(y, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

# Прогноз 10 периодов
forecast_auto = auto_model.predict(n_periods=10)
plt.figure()
plt.plot(np.arange(len(y)), y, label='Original')
plt.plot(np.arange(len(y), len(y)+10), forecast_auto, label='Forecast')
plt.title("Optimal ARIMA Forecast")
plt.legend()
plt.show()

```


Задание 2

<img width="803" height="588" alt="image" src="https://github.com/user-attachments/assets/855b96e7-1472-449e-ab97-e779cec92b14" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- логарифм US M2 (\textbf{месячные данные}) с 1995 по н.в.
\begin{enumerate}
	\item Подгонка модели заданного порядка
	\begin{enumerate}
		\item Подгоните модели
		\begin{center}
		\begin{tabular}{l|c}
			Модель & drift/trend \\ \hline
			ARIMA(2,0,2) & + \\
			ARIMA(2,1,0) & + \\
			ARIMA(2,1,1) & - \\
			ARIMA(1,2,0) & - \\ \hline
		\end{tabular}
		\end{center} 
		и постройте прогноз на 10 периодов. Значим ли снос/тренд?
		\item Проведите диагностику каждой модели.
		\item Проведите кросс-валидацию каждой модели. Какая предпочтительней?
	\end{enumerate}
	\item Примените тесты единичного корня и найдите порядок интегрирования для \(y_t\). 
	\item Подгонка <<оптимальной модели>>
	\begin{enumerate}
		\item Подгоните <<оптимальную>> модель ARIMA
		\item проведите её диагностику
		\item Постройте прогноз на 10 периодов
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Анализ US M2: ARIMA подгонка, прогноз, тесты единичного корня
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader.data as web

plt.style.use('ggplot')

# -----------------------------
# 1. Загрузка данных
# -----------------------------
m2 = web.DataReader('M2SL', data_source='fred', start='1995-01-01', end='2023-12-31')
y = np.log(m2['M2SL'])  # логарифм ряда
y.index = y.index.to_period('M')  # месячный временной индекс

y.plot(title='Log of US M2')
plt.show()

# -----------------------------
# 2. Подгонка моделей ARIMA
# -----------------------------
# ARIMA(2,0,2) с drift/trend
model_202 = ARIMA(y, order=(2,0,2), trend='ct')
res_202 = model_202.fit()
print(res_202.summary())
res_202.get_forecast(steps=10).predicted_mean.plot(title='Forecast ARIMA(2,0,2)')
plt.show()

# ARIMA(2,1,0) с drift
model_210 = ARIMA(y, order=(2,1,0), trend='c')
res_210 = model_210.fit()
print(res_210.summary())
res_210.get_forecast(steps=10).predicted_mean.plot(title='Forecast ARIMA(2,1,0)')
plt.show()

# ARIMA(2,1,1) без drift/trend
model_211 = ARIMA(y, order=(2,1,1), trend='n')
res_211 = model_211.fit()
print(res_211.summary())
res_211.get_forecast(steps=10).predicted_mean.plot(title='Forecast ARIMA(2,1,1)')
plt.show()

# ARIMA(1,2,0) без drift/trend
model_120 = ARIMA(y, order=(1,2,0), trend='n')
res_120 = model_120.fit()
print(res_120.summary())
res_120.get_forecast(steps=10).predicted_mean.plot(title='Forecast ARIMA(1,2,0)')
plt.show()

# -----------------------------
# 3. Диагностика моделей
# -----------------------------
for res, name in zip([res_202, res_210, res_211, res_120],
                     ['ARIMA(2,0,2)','ARIMA(2,1,0)','ARIMA(2,1,1)','ARIMA(1,2,0)']):
    print(f"=== Диагностика {name} ===")
    res.plot_diagnostics(figsize=(10,8))
    plt.show()

# -----------------------------
# 4. Кросс-валидация
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
models = {'ARIMA202': (2,0,2), 'ARIMA210': (2,1,0), 'ARIMA211': (2,1,1), 'ARIMA120': (1,2,0)}

for name, order in models.items():
    errors = []
    for train_idx, test_idx in tscv.split(y):
        train, test = y.iloc[train_idx], y.iloc[test_idx]
        trend = 'ct' if order==(2,0,2) else 'c' if order==(2,1,0) else 'n'
        model = ARIMA(train, order=order, trend=trend).fit()
        pred = model.forecast(len(test))
        errors.append(np.mean((pred - test)**2))
    print(f'{name} CV MSE: {np.mean(errors)}')

# -----------------------------
# 5. Тест единичного корня (ADF)
# -----------------------------
adf_result = adfuller(y)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
if adf_result[1] < 0.05:
    print("Ряд стационарен")
else:
    print("Ряд нестационарен, нужен порядок дифференцирования")

# -----------------------------
# 6. Автоматическая оптимальная ARIMA
# -----------------------------
auto_model = auto_arima(y, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

forecast_auto = auto_model.predict(n_periods=10)
plt.figure()
plt.plot(np.arange(len(y)), y, label='Original')
plt.plot(np.arange(len(y), len(y)+10), forecast_auto, label='Forecast')
plt.title("Optimal ARIMA Forecast")
plt.legend()
plt.show()

```

Задание 3

<img width="752" height="1066" alt="image" src="https://github.com/user-attachments/assets/554a98ca-b9ff-42a2-8bef-dc62340d7a7b" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- логарифм US M2 (\textbf{недельные данные}) с 1995 по н.в.
\begin{enumerate}
	\item Подгонка модели заданного порядка
	\begin{enumerate}
		\item Подгоните модели
		\begin{center}
		\begin{tabular}{l|c}
			Модель & drift/trend \\ \hline
			ARIMA(3,0,2) & + \\
			ARIMA(2,1,0) & + \\
			ARIMA(2,1,1) & - \\
			ARIMA(2,2,0) & - \\ \hline
		\end{tabular}
		\end{center} 
		и постройте прогноз на 10 периодов. Значим ли снос/тренд?
		\item Проведите диагностику каждой модели.
		\item Проведите кросс-валидацию каждой модели. Какая предпочтительней?
	\end{enumerate}
	\item Примените тесты единичного корня и найдите порядок интегрирования для \(y_t\). 
	\item Подгонка <<оптимальной модели>>
	\begin{enumerate}
		\item Подгоните <<оптимальную>> модель ARIMA
		\item проведите её диагностику
		\item Постройте прогноз на 10 периодов
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Анализ US M2: ARIMA подгонка, прогноз, тесты единичного корня (недельные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader.data as web

plt.style.use('ggplot')

# -----------------------------
# 1. Загрузка данных (недельные)
# -----------------------------
m2_weekly = web.DataReader('M2SL', data_source='fred', start='1995-01-01', end='2023-12-31')
y = np.log(m2_weekly['M2SL'])
y.index = y.index.to_period('W')  # недельный временной индекс

y.plot(title='Log of US M2 (Weekly)')
plt.show()

# -----------------------------
# 2. Подгонка моделей ARIMA
# -----------------------------
models_order = {
    'ARIMA302': (3,0,2,'ct'),
    'ARIMA210': (2,1,0,'c'),
    'ARIMA211': (2,1,1,'n'),
    'ARIMA220': (2,2,0,'n')
}

results = {}
for name, (p,d,q,trend) in models_order.items():
    model = ARIMA(y, order=(p,d,q), trend=trend)
    res = model.fit()
    results[name] = res
    print(f"=== {name} ===")
    print(res.summary())
    # Прогноз на 10 недель
    forecast = res.get_forecast(steps=10).predicted_mean
    plt.figure()
    plt.plot(y.index, y, label='Original')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title(f"{name} Forecast")
    plt.legend()
    plt.show()

# -----------------------------
# 3. Диагностика моделей
# -----------------------------
for name, res in results.items():
    print(f"=== Диагностика {name} ===")
    res.plot_diagnostics(figsize=(10,8))
    plt.show()

# -----------------------------
# 4. Кросс-валидация
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
for name, (p,d,q,trend) in models_order.items():
    errors = []
    for train_idx, test_idx in tscv.split(y):
        train, test = y.iloc[train_idx], y.iloc[test_idx]
        res = ARIMA(train, order=(p,d,q), trend=trend).fit()
        pred = res.forecast(len(test))
        errors.append(np.mean((pred - test)**2))
    print(f'{name} CV MSE: {np.mean(errors)}')

# -----------------------------
# 5. Тест единичного корня (ADF)
# -----------------------------
adf_result = adfuller(y)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
if adf_result[1] < 0.05:
    print("Ряд стационарен")
else:
    print("Ряд нестационарен, нужен порядок дифференцирования")

# -----------------------------
# 6. Автоматическая оптимальная ARIMA
# -----------------------------
auto_model = auto_arima(y, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

forecast_auto = auto_model.predict(n_periods=10)
plt.figure()
plt.plot(y.index, y, label='Original')
plt.plot(pd.period_range(start=y.index[-1]+1, periods=10, freq='W'), forecast_auto, label='Forecast')
plt.title("Optimal ARIMA Forecast")
plt.legend()
plt.show()

```


Задание 4

<img width="787" height="1084" alt="image" src="https://github.com/user-attachments/assets/b89b1837-1081-4f8a-80bd-7b50cc57741a" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- 10-летняя ставка (treasury securities  with constant maturity 
\textbf{месячные данные}) с 2000 по н.в.
\begin{enumerate}
	\item Подгонка модели заданного порядка
	\begin{enumerate}
		\item Подгоните модели
		\begin{center}
		\begin{tabular}{l|c}
			Модель & drift/const \\ \hline
			ARIMA(2,0,2) & - \\
			ARIMA(2,0,2) & + \\
			ARIMA(2,1,0) & + \\
			ARIMA(2,1,1) & - \\
			ARIMA(2,2,0) & - \\ \hline
		\end{tabular}
		\end{center} 
		и постройте прогноз на 10 периодов. Значим ли снос/const?
		\item Проведите диагностику каждой модели.
		\item Проведите кросс-валидацию каждой модели. Какая предпочтительней?
	\end{enumerate}
	\item Примените тесты единичного корня и найдите порядок интегрирования для \(y_t\). 
	\item Подгонка <<оптимальной модели>>
	\begin{enumerate}
		\item Подгоните <<оптимальную>> модель ARIMA
		\item проведите её диагностику
		\item Постройте прогноз на 10 периодов
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Анализ 10-летней ставки: ARIMA подгонка, прогноз, тесты единичного корня (месячные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader.data as web

plt.style.use('ggplot')

# -----------------------------
# 1. Загрузка месячных данных 10-летних казначейских облигаций
# -----------------------------
rate10 = web.DataReader('GS10', data_source='fred', start='2000-01-01', end='2023-12-31')
y = rate10['GS10']
y.index = y.index.to_period('M')  # месячный временной индекс

y.plot(title='10-Year Treasury Yield (Monthly)')
plt.show()

# -----------------------------
# 2. Подгонка моделей ARIMA
# -----------------------------
models_order = {
    'ARIMA202_nodrift': (2,0,2,'n'),  # без константы
    'ARIMA202_const': (2,0,2,'c'),    # с константой
    'ARIMA210_const': (2,1,0,'c'),
    'ARIMA211_nodrift': (2,1,1,'n'),
    'ARIMA220_nodrift': (2,2,0,'n')
}

results = {}
for name, (p,d,q,trend) in models_order.items():
    model = ARIMA(y, order=(p,d,q), trend=trend)
    res = model.fit()
    results[name] = res
    print(f"=== {name} ===")
    print(res.summary())
    # Прогноз на 10 месяцев
    forecast = res.get_forecast(steps=10).predicted_mean
    plt.figure()
    plt.plot(y.index, y, label='Original')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title(f"{name} Forecast")
    plt.legend()
    plt.show()

# -----------------------------
# 3. Диагностика моделей
# -----------------------------
for name, res in results.items():
    print(f"=== Диагностика {name} ===")
    res.plot_diagnostics(figsize=(10,8))
    plt.show()

# -----------------------------
# 4. Кросс-валидация
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
for name, (p,d,q,trend) in models_order.items():
    errors = []
    for train_idx, test_idx in tscv.split(y):
        train, test = y.iloc[train_idx], y.iloc[test_idx]
        res = ARIMA(train, order=(p,d,q), trend=trend).fit()
        pred = res.forecast(len(test))
        errors.append(np.mean((pred - test)**2))
    print(f'{name} CV MSE: {np.mean(errors)}')

# -----------------------------
# 5. Тест единичного корня (ADF)
# -----------------------------
adf_result = adfuller(y)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
if adf_result[1] < 0.05:
    print("Ряд стационарен")
else:
    print("Ряд нестационарен, нужен порядок дифференцирования")

# -----------------------------
# 6. Автоматическая оптимальная ARIMA
# -----------------------------
auto_model = auto_arima(y, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

forecast_auto = auto_model.predict(n_periods=10)
plt.figure()
plt.plot(y.index, y, label='Original')
plt.plot(pd.period_range(start=y.index[-1]+1, periods=10, freq='M'), forecast_auto, label='Forecast')
plt.title("Optimal ARIMA Forecast")
plt.legend()
plt.show()

```


Задание 5

<img width="767" height="659" alt="image" src="https://github.com/user-attachments/assets/ee320d43-55ab-4e3a-b18e-b1d0f720d39a" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- 10-летняя ставка (treasury securities with constant matu\-ri\-ty) 
(\textbf{дневные данные}) с 2010 по н.в.
\begin{enumerate}
	\item Подгонка модели заданного порядка
	\begin{enumerate}
		\item Подгоните модели
		\begin{center}
		\begin{tabular}{l|c}
			Модель & drift/const \\ \hline
			ARIMA(3,0,2) & - \\
			ARIMA(3,0,2) & + \\
			ARIMA(3,1,0) & + \\
			ARIMA(3,1,1) & - \\
			ARIMA(2,2,0) & - \\ \hline
		\end{tabular}
		\end{center} 
		и постройте прогноз на 10 периодов. Значим ли снос/const?
		\item Проведите диагностику каждой модели.
		\item Проведите кросс-валидацию каждой модели. Какая предпочтительней?
	\end{enumerate}
	\item Примените тесты единичного корня и найдите порядок интегрирования для \(y_t\). 
	\item Подгонка <<оптимальной модели>>
	\begin{enumerate}
		\item Подгоните <<оптимальную>> модель ARIMA
		\item проведите её диагностику
		\item Постройте прогноз на 10 периодов
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Анализ 10-летней ставки: ARIMA подгонка, прогноз, тесты единичного корня (дневные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader.data as web

plt.style.use('ggplot')

# -----------------------------
# 1. Загрузка дневных данных 10-летних казначейских облигаций
# -----------------------------
rate10_daily = web.DataReader('GS10', data_source='fred', start='2010-01-01', end='2023-12-31')
y = rate10_daily['GS10']
y.index = pd.to_datetime(y.index)

y.plot(title='10-Year Treasury Yield (Daily)')
plt.show()

# -----------------------------
# 2. Подгонка моделей ARIMA
# -----------------------------
models_order = {
    'ARIMA302_nodrift': (3,0,2,'n'),  # без константы
    'ARIMA302_const': (3,0,2,'c'),    # с константой
    'ARIMA310_const': (3,1,0,'c'),
    'ARIMA311_nodrift': (3,1,1,'n'),
    'ARIMA220_nodrift': (2,2,0,'n')
}

results = {}
for name, (p,d,q,trend) in models_order.items():
    model = ARIMA(y, order=(p,d,q), trend=trend)
    res = model.fit()
    results[name] = res
    print(f"=== {name} ===")
    print(res.summary())
    # Прогноз на 10 дней
    forecast = res.get_forecast(steps=10).predicted_mean
    plt.figure()
    plt.plot(y.index, y, label='Original')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title(f"{name} Forecast")
    plt.legend()
    plt.show()

# -----------------------------
# 3. Диагностика моделей
# -----------------------------
for name, res in results.items():
    print(f"=== Диагностика {name} ===")
    res.plot_diagnostics(figsize=(10,8))
    plt.show()

# -----------------------------
# 4. Кросс-валидация
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
for name, (p,d,q,trend) in models_order.items():
    errors = []
    for train_idx, test_idx in tscv.split(y):
        train, test = y.iloc[train_idx], y.iloc[test_idx]
        res = ARIMA(train, order=(p,d,q), trend=trend).fit()
        pred = res.forecast(len(test))
        errors.append(np.mean((pred - test)**2))
    print(f'{name} CV MSE: {np.mean(errors)}')

# -----------------------------
# 5. Тест единичного корня (ADF)
# -----------------------------
adf_result = adfuller(y)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
if adf_result[1] < 0.05:
    print("Ряд стационарен")
else:
    print("Ряд нестационарен, нужен порядок дифференцирования")

# -----------------------------
# 6. Автоматическая оптимальная ARIMA
# -----------------------------
auto_model = auto_arima(y, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

forecast_auto = auto_model.predict(n_periods=10)
plt.figure()
plt.plot(y.index, y, label='Original')
plt.plot(pd.date_range(start=y.index[-1]+pd.Timedelta(days=1), periods=10, freq='B'), forecast_auto, label='Forecast')
plt.title("Optimal ARIMA Forecast")
plt.legend()
plt.show()

```


Задание 6

<img width="751" height="1067" alt="image" src="https://github.com/user-attachments/assets/13927853-e8d6-49b5-8dcb-0158646f8530" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- 3-месячная ставки (treasury bill, \textbf{месячные данные}) с 2000 по н.в.
\begin{enumerate}
	\item Подгонка модели заданного порядка
	\begin{enumerate}
		\item Подгоните модели
		\begin{center}
		\begin{tabular}{l|c}
			Модель & drift/const \\ \hline
			ARIMA(2,0,2) & - \\
			ARIMA(2,0,2) & + \\
			ARIMA(2,1,0) & + \\
			ARIMA(2,1,1) & - \\
			ARIMA(2,2,0) & - \\ \hline
		\end{tabular}
		\end{center} 
		и постройте прогноз на 10 периодов. Значим ли снос/const?
		\item Проведите диагностику каждой модели.
		\item Проведите кросс-валидацию каждой модели. Какая предпочтительней?
	\end{enumerate}
	\item Примените тесты единичного корня и найдите порядок интегрирования для \(y_t\). 
	\item Подгонка <<оптимальной модели>>
	\begin{enumerate}
		\item Подгоните <<оптимальную>> модель ARIMA
		\item проведите её диагностику
		\item Постройте прогноз на 10 периодов
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Анализ 3-месячной ставки (месячные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader.data as web

plt.style.use('ggplot')

# -----------------------------
# 1. Загрузка данных 3-месячных казначейских облигаций
# -----------------------------
rate3m = web.DataReader('TB3MS', data_source='fred', start='2000-01-01', end='2023-12-31')
y = rate3m['TB3MS']
y.index = pd.to_datetime(y.index)

y.plot(title='3-Month Treasury Bill Yield (Monthly)')
plt.show()

# -----------------------------
# 2. Подгонка моделей ARIMA
# -----------------------------
models_order = {
    'ARIMA202_nodrift': (2,0,2,'n'),  # без константы
    'ARIMA202_const': (2,0,2,'c'),    # с константой
    'ARIMA210_const': (2,1,0,'c'),
    'ARIMA211_nodrift': (2,1,1,'n'),
    'ARIMA220_nodrift': (2,2,0,'n')
}

results = {}
for name, (p,d,q,trend) in models_order.items():
    model = ARIMA(y, order=(p,d,q), trend=trend)
    res = model.fit()
    results[name] = res
    print(f"=== {name} ===")
    print(res.summary())
    
    # Прогноз на 10 месяцев
    forecast = res.get_forecast(steps=10).predicted_mean
    plt.figure()
    plt.plot(y.index, y, label='Original')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title(f"{name} Forecast")
    plt.legend()
    plt.show()

# -----------------------------
# 3. Диагностика моделей
# -----------------------------
for name, res in results.items():
    print(f"=== Диагностика {name} ===")
    res.plot_diagnostics(figsize=(10,8))
    plt.show()

# -----------------------------
# 4. Кросс-валидация
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
for name, (p,d,q,trend) in models_order.items():
    errors = []
    for train_idx, test_idx in tscv.split(y):
        train, test = y.iloc[train_idx], y.iloc[test_idx]
        res = ARIMA(train, order=(p,d,q), trend=trend).fit()
        pred = res.forecast(len(test))
        errors.append(np.mean((pred - test)**2))
    print(f'{name} CV MSE: {np.mean(errors)}')

# -----------------------------
# 5. Тест единичного корня (ADF)
# -----------------------------
adf_result = adfuller(y)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
if adf_result[1] < 0.05:
    print("Ряд стационарен")
else:
    print("Ряд нестационарен, нужен порядок дифференцирования")

# -----------------------------
# 6. Автоматическая оптимальная ARIMA
# -----------------------------
auto_model = auto_arima(y, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

forecast_auto = auto_model.predict(n_periods=10)
plt.figure()
plt.plot(y.index, y, label='Original')
plt.plot(pd.date_range(start=y.index[-1]+pd.DateOffset(months=1), periods=10, freq='M'), forecast_auto, label='Forecast')
plt.title("Optimal ARIMA Forecast")
plt.legend()
plt.show()

```


Задание 7

<img width="754" height="1070" alt="image" src="https://github.com/user-attachments/assets/2444b618-f5c8-4b40-9641-df8b747cbf31" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- 3-месячная ставки (treasury bill, \textbf{дневные данные}) с 2010 по н.в.
\begin{enumerate}
	\item Подгонка модели заданного порядка
	\begin{enumerate}
		\item Подгоните модели
		\begin{center}
		\begin{tabular}{l|c}
			Модель & drift/const \\ \hline
			ARIMA(3,0,2) & - \\
			ARIMA(3,0,2) & + \\
			ARIMA(3,1,0) & + \\
			ARIMA(3,1,1) & - \\
			ARIMA(2,2,0) & - \\ \hline
		\end{tabular}
		\end{center} 
		и постройте прогноз на 10 периодов. Значим ли снос/const?
		\item Проведите диагностику каждой модели.
		\item Проведите кросс-валидацию каждой модели. Какая предпочтительней?
	\end{enumerate}
	\item Примените тесты единичного корня и найдите порядок интегрирования для \(y_t\). 
	\item Подгонка <<оптимальной модели>>
	\begin{enumerate}
		\item Подгоните <<оптимальную>> модель ARIMA
		\item проведите её диагностику
		\item Постройте прогноз на 10 периодов
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# ======================================================
# Анализ 3-месячной ставки (дневные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader.data as web

plt.style.use('ggplot')

# -----------------------------
# 1. Загрузка данных 3-месячных казначейских облигаций (дневные)
# -----------------------------
rate3m_daily = web.DataReader('TB3MS', data_source='fred', start='2010-01-01', end='2023-12-31')
y = rate3m_daily['TB3MS'].dropna()  # удаляем пропуски
y.index = pd.to_datetime(y.index)

y.plot(title='3-Month Treasury Bill Yield (Daily)')
plt.show()

# -----------------------------
# 2. Подгонка моделей ARIMA
# -----------------------------
models_order = {
    'ARIMA302_nodrift': (3,0,2,'n'),  # без константы
    'ARIMA302_const': (3,0,2,'c'),    # с константой
    'ARIMA310_const': (3,1,0,'c'),
    'ARIMA311_nodrift': (3,1,1,'n'),
    'ARIMA220_nodrift': (2,2,0,'n')
}

results = {}
for name, (p,d,q,trend) in models_order.items():
    model = ARIMA(y, order=(p,d,q), trend=trend)
    res = model.fit()
    results[name] = res
    print(f"=== {name} ===")
    print(res.summary())
    
    # Прогноз на 10 дней
    forecast = res.get_forecast(steps=10).predicted_mean
    plt.figure()
    plt.plot(y.index, y, label='Original')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title(f"{name} Forecast")
    plt.legend()
    plt.show()

# -----------------------------
# 3. Диагностика моделей
# -----------------------------
for name, res in results.items():
    print(f"=== Диагностика {name} ===")
    res.plot_diagnostics(figsize=(10,8))
    plt.show()

# -----------------------------
# 4. Кросс-валидация
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
for name, (p,d,q,trend) in models_order.items():
    errors = []
    for train_idx, test_idx in tscv.split(y):
        train, test = y.iloc[train_idx], y.iloc[test_idx]
        res = ARIMA(train, order=(p,d,q), trend=trend).fit()
        pred = res.forecast(len(test))
        errors.append(np.mean((pred - test)**2))
    print(f'{name} CV MSE: {np.mean(errors)}')

# -----------------------------
# 5. Тест единичного корня (ADF)
# -----------------------------
adf_result = adfuller(y)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
if adf_result[1] < 0.05:
    print("Ряд стационарен")
else:
    print("Ряд нестационарен, нужен порядок дифференцирования")

# -----------------------------
# 6. Автоматическая оптимальная ARIMA
# -----------------------------
auto_model = auto_arima(y, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

forecast_auto = auto_model.predict(n_periods=10)
plt.figure()
plt.plot(y.index, y, label='Original')
plt.plot(pd.date_range(start=y.index[-1]+pd.Timedelta(days=1), periods=10, freq='B'), forecast_auto, label='Forecast')
plt.title("Optimal ARIMA Forecast")
plt.legend()
plt.show()

```


Задание 8

<img width="759" height="624" alt="image" src="https://github.com/user-attachments/assets/cd0185ea-9481-4b6f-a2d4-0e66808a502c" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- логарифм S\&P500 (\textbf{дневные данные}) с 2010 по н.в.
\begin{enumerate}
	\item Подгонка модели заданного порядка
		\begin{enumerate}
			\item Подгоните модели
			\begin{center}
			\begin{tabular}{l|c}
				Модель & drift/const \\ \hline
				ARIMA(2,0,2) & - \\
				ARIMA(2,0,2) & + \\
				ARIMA(2,1,0) & + \\
				ARIMA(2,1,1) & - \\
				ARIMA(2,2,0) & - \\ \hline
			\end{tabular}
			\end{center} 
			и постройте прогноз на 10 периодов. Значим ли снос/const?
			\item Проведите диагностику каждой модели.
			\item Проведите кросс-валидацию каждой модели. Какая предпочтительней?
		\end{enumerate}
		\item Примените тесты единичного корня и найдите порядок интегрирования для \(y_t\). 
		\item Подгонка <<оптимальной модели>>
		\begin{enumerate}
			\item Подгоните <<оптимальную>> модель ARIMA
			\item проведите её диагностику
			\item Постройте прогноз на 10 периодов
		\end{enumerate}
	\end{enumerate}
	\end{exercise}
```


Ответ

```
# ======================================================
# Анализ логарифма S&P500 (дневные данные)
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf

plt.style.use('ggplot')

# -----------------------------
# 1. Загрузка данных S&P500
# -----------------------------
sp500 = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')['Adj Close']
y = np.log(sp500).dropna()
y.plot(title='Log S&P500 (Daily)')
plt.show()

# -----------------------------
# 2. Подгонка моделей ARIMA
# -----------------------------
models_order = {
    'ARIMA202_nodrift': (2,0,2,'n'),  # без константы
    'ARIMA202_const': (2,0,2,'c'),    # с константой
    'ARIMA210_const': (2,1,0,'c'),
    'ARIMA211_nodrift': (2,1,1,'n'),
    'ARIMA220_nodrift': (2,2,0,'n')
}

results = {}
for name, (p,d,q,trend) in models_order.items():
    model = ARIMA(y, order=(p,d,q), trend=trend)
    res = model.fit()
    results[name] = res
    print(f"=== {name} ===")
    print(res.summary())
    
    # Прогноз на 10 дней
    forecast = res.get_forecast(steps=10).predicted_mean
    plt.figure()
    plt.plot(y.index, y, label='Original')
    plt.plot(forecast.index, forecast, label='Forecast')
    plt.title(f"{name} Forecast")
    plt.legend()
    plt.show()

# -----------------------------
# 3. Диагностика моделей
# -----------------------------
for name, res in results.items():
    print(f"=== Диагностика {name} ===")
    res.plot_diagnostics(figsize=(10,8))
    plt.show()

# -----------------------------
# 4. Кросс-валидация
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
for name, (p,d,q,trend) in models_order.items():
    errors = []
    for train_idx, test_idx in tscv.split(y):
        train, test = y.iloc[train_idx], y.iloc[test_idx]
        res = ARIMA(train, order=(p,d,q), trend=trend).fit()
        pred = res.forecast(len(test))
        errors.append(np.mean((pred - test)**2))
    print(f'{name} CV MSE: {np.mean(errors)}')

# -----------------------------
# 5. Тест единичного корня (ADF)
# -----------------------------
adf_result = adfuller(y)
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
if adf_result[1] < 0.05:
    print("Ряд стационарен")
else:
    print("Ряд нестационарен, нужен порядок дифференцирования")

# -----------------------------
# 6. Автоматическая оптимальная ARIMA
# -----------------------------
auto_model = auto_arima(y, seasonal=False, stepwise=True, trace=True)
print(auto_model.summary())

forecast_auto = auto_model.predict(n_periods=10)
plt.figure()
plt.plot(y.index, y, label='Original')
plt.plot(pd.date_range(start=y.index[-1]+pd.Timedelta(days=1), periods=10, freq='B'), forecast_auto, label='Forecast')
plt.title("Optimal ARIMA Forecast")
plt.legend()
plt.show()

```


Задание 9


<img width="542" height="345" alt="image" src="https://github.com/user-attachments/assets/3b1843ac-3fda-4b23-bb94-b54cad62e042" />


Текст

```
\begin{exercise}
Запишите спецификации следующих моделей
\begin{enumerate}
	\item ARIMA(0,1,1) без сноса и со сносом
	\item ARIMA(0,1,2) без сноса и со сносом
	\item ARIMA(1,1,0) без сноса и со сносом
	\item ARIMA(2,1,0) без сноса и со сносом
	\item ARIMA(0,2,0) без сноса и со сносом
	\item ARIMA(1,2,0) без сноса и со сносом
	\item ARIMA(0,2,1) без сноса и со сносом
\end{enumerate}
\end{exercise}
```


Ответ

```
# Импортируем нужные библиотеки
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Допустим, y — твой временной ряд (например, log(GDP) или log(M2))
# y = pd.Series(...)

# Словарь с моделями: ключ = название модели, значение = (p,d,q, снос/const)
models = {
    "ARIMA(0,1,1)_no_drift": (0,1,1, False),
    "ARIMA(0,1,1)_with_drift": (0,1,1, True),
    "ARIMA(0,1,2)_no_drift": (0,1,2, False),
    "ARIMA(0,1,2)_with_drift": (0,1,2, True),
    "ARIMA(1,1,0)_no_drift": (1,1,0, False),
    "ARIMA(1,1,0)_with_drift": (1,1,0, True),
    "ARIMA(2,1,0)_no_drift": (2,1,0, False),
    "ARIMA(2,1,0)_with_drift": (2,1,0, True),
    "ARIMA(0,2,0)_no_drift": (0,2,0, False),
    "ARIMA(0,2,0)_with_drift": (0,2,0, True),
    "ARIMA(1,2,0)_no_drift": (1,2,0, False),
    "ARIMA(1,2,0)_with_drift": (1,2,0, True),
    "ARIMA(0,2,1)_no_drift": (0,2,1, False),
    "ARIMA(0,2,1)_with_drift": (0,2,1, True),
}

# Пример подгонки всех моделей
results = {}
for name, (p,d,q,trend) in models.items():
    # Подгонка модели
    model = ARIMA(y, order=(p,d,q), trend='c' if trend else 'n')
    fit = model.fit()
    
    # Сохраняем результат
    results[name] = fit
    print(f"{name}:\n", fit.summary(), "\n")

```

4 Модель (*)ARCH

Задание 1

<img width="766" height="1161" alt="image" src="https://github.com/user-attachments/assets/707562ae-3310-4a60-ab3c-5bfccbdd83d3" />


Текст

```
\begin{exercise}
Пусть \(y_t\) -- лог-доходность US M2 (\textbf{недельные данные}) с 1995 по н.в.
\begin{enumerate}
	\item Подгоните модели AR-GARCH(p,o,q)
	\begin{center}
	\begin{tabular}{l|c}
		Модель & \(\lambda\) \\ \hline
		AR(1)-GARCH(1,0,1) & 2 \\
		AR(1)-GARCH(1,0,1) & 1 \\
		AR(1)-GARCH(1,1,1) & 2 \\
		AR(2)-GARCH(1,0,1) & 2 \\
		AR(2)-GARCH(1,0,1) & 1 \\ \hline
	\end{tabular}
	\end{center} 
	и постройте прогноз на 10 периодов для ряда и его волатильности.
	\item Сравните модели по информационным критериям. Какая предпочтительней?
	\item Проведите кросс-валидацию моделей. Какая предпочтительней?
	\item Подгоните модели
	\begin{center}
		\begin{tabular}{l}
			Модель  \\ \hline
			AR(1)-EGARCH(1,0,1) \\
			AR(1)-EGARCH(1,1,1) \\
			AR(1)-APARCH(1,0,1) \\
			AR(1)-APARCH(1,1,1)
		\end{tabular}
	\end{center} 
	и постройте прогноз на 10 периодов для ряда и его волатильности.
	\item Какая предпочтительней?
\end{enumerate}
\end{exercise}
```


Ответ

```
# Импортируем библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Предположим, y — это лог-доходность US M2 (еженедельные данные)
# y = pd.Series(..., index=pd.date_range(...))

# -------------------------------
# 1. Подгонка AR-GARCH(p,o,q) моделей
# -------------------------------

# Список моделей
ar_garch_models = [
    {"ar":1, "p":1, "o":0, "q":1, "lambda":2},
    {"ar":1, "p":1, "o":0, "q":1, "lambda":1},
    {"ar":1, "p":1, "o":1, "q":1, "lambda":2},
    {"ar":2, "p":1, "o":0, "q":1, "lambda":2},
    {"ar":2, "p":1, "o":0, "q":1, "lambda":1},
]

# Словарь для хранения результатов
results_ar_garch = {}

for m in ar_garch_models:
    am = arch_model(y, mean='AR', lags=m["ar"],
                    vol='GARCH', p=m["p"], o=m["o"], q=m["q"], power=m["lambda"],
                    dist='normal')
    res = am.fit(disp='off')
    results_ar_garch[f'AR({m["ar"]})-GARCH({m["p"]},{m["o"]},{m["q"]})_lambda{m["lambda"]}'] = res
    print(res.summary())
    
    # Прогноз на 10 периодов вперед
    forecast = res.forecast(horizon=10)
    print("Прогноз условной волатильности:\n", forecast.variance[-1:])

# -------------------------------
# 2. Сравнение по информационным критериям
# -------------------------------
for name, res in results_ar_garch.items():
    print(name, "AIC:", res.aic, "BIC:", res.bic)

# -------------------------------
# 3. Кросс-валидация
# -------------------------------
# Можно реализовать с помощью rolling forecast или walk-forward validation
# Пример:
from sklearn.metrics import mean_squared_error

rolling_window = 200  # пример: использовать последние 200 недель для обучения
forecast_horizon = 1
mses = {}

for name, m in zip(results_ar_garch.keys(), ar_garch_models):
    errors = []
    for i in range(rolling_window, len(y)-forecast_horizon):
        train = y[i-rolling_window:i]
        model = arch_model(train, mean='AR', lags=m["ar"],
                           vol='GARCH', p=m["p"], o=m["o"], q=m["q"], power=m["lambda"])
        fit = model.fit(disp='off')
        pred = fit.forecast(horizon=forecast_horizon).mean.iloc[-1,0]
        errors.append((y[i] - pred)**2)
    mses[name] = np.mean(errors)

best_model_name = min(mses, key=mses.get)
print("Лучший вариант по кросс-валидации:", best_model_name)

# -------------------------------
# 4. Подгонка AR-EGARCH и AR-APARCH
# -------------------------------
egarch_aparch_models = [
    {"vol":"EGARCH", "p":1, "o":0, "q":1, "ar":1},
    {"vol":"EGARCH", "p":1, "o":1, "q":1, "ar":1},
    {"vol":"APARCH", "p":1, "o":0, "q":1, "ar":1},
    {"vol":"APARCH", "p":1, "o":1, "q":1, "ar":1},
]

results_egarch_aparch = {}

for m in egarch_aparch_models:
    am = arch_model(y, mean='AR', lags=m["ar"], vol=m["vol"], p=m["p"], o=m["o"], q=m["q"], dist='normal')
    res = am.fit(disp='off')
    results_egarch_aparch[f'AR({m["ar"]})-{m["vol"]}({m["p"]},{m["o"]},{m["q"]})'] = res
    print(res.summary())
    
    forecast = res.forecast(horizon=10)
    print("Прогноз условной волатильности:\n", forecast.variance[-1:])

# Сравнение по AIC/BIC и выбор наилучшей модели
for name, res in results_egarch_aparch.items():
    print(name, "AIC:", res.aic, "BIC:", res.bic)

```

Задание 2

<img width="735" height="713" alt="image" src="https://github.com/user-attachments/assets/041c7ba1-2c9f-4c74-abe7-dc28a057cfeb" />

Текст

```
\begin{exercise}
Пусть ряд \(y_t\) -- первая разность 3-месячной ставки (treasury bill, \textbf{дневные данные}) с 2010 по н.в.
\begin{enumerate}
	\item Подгоните модели AR-GARCH(p,o,q)
	\begin{center}
	\begin{tabular}{l|c}
		Модель & \(\lambda\) \\ \hline
		AR(1)-GARCH(1,0,1) & 2 \\
		AR(1)-GARCH(1,0,1) & 1 \\
		AR(1)-GARCH(1,1,1) & 2 \\
		AR(2)-GARCH(1,0,1) & 2 \\
		AR(2)-GARCH(1,0,1) & 1 \\ \hline
	\end{tabular}
	\end{center} 
	и постройте прогноз на 10 периодов для ряда и его волатильности.
	\item Сравните модели по информационным критериям. Какая предпочтительней?
	\item Проведите кросс-валидацию моделей. Какая предпочтительней?
	\item Подгоните модели
	\begin{center}
		\begin{tabular}{l}
			Модель  \\ \hline
			AR(1)-EGARCH(1,0,1) \\
			AR(1)-EGARCH(1,1,1) \\
			AR(1)-APARCH(1,0,1) \\
			AR(1)-APARCH(1,1,1)
		\end{tabular}
	\end{center} 
	и постройте прогноз на 10 периодов для ряда и его волатильности.
	\item Какая предпочтительней?
\end{enumerate}
\end{exercise}
```

Ответы

```
# -------------------------------
# Импорт библиотек
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# Предположим, y — это первая разность 3-месячной ставки (дневные данные)
# y = pd.Series(..., index=pd.date_range(...))

# -------------------------------
# 1. Подгонка AR-GARCH(p,o,q) моделей
# -------------------------------
ar_garch_models = [
    {"ar":1, "p":1, "o":0, "q":1, "lambda":2},
    {"ar":1, "p":1, "o":0, "q":1, "lambda":1},
    {"ar":1, "p":1, "o":1, "q":1, "lambda":2},
    {"ar":2, "p":1, "o":0, "q":1, "lambda":2},
    {"ar":2, "p":1, "o":0, "q":1, "lambda":1},
]

results_ar_garch = {}
for m in ar_garch_models:
    model_name = f'AR({m["ar"]})-GARCH({m["p"]},{m["o"]},{m["q"]})_lambda{m["lambda"]}'
    am = arch_model(y, mean='AR', lags=m["ar"], 
                    vol='GARCH', p=m["p"], o=m["o"], q=m["q"], power=m["lambda"],
                    dist='normal')
    res = am.fit(disp='off')
    results_ar_garch[model_name] = res
    print(res.summary())
    
    # Прогноз на 10 периодов вперед
    forecast = res.forecast(horizon=10)
    print(f"Прогноз волатильности для {model_name}:\n", forecast.variance[-1:])

# -------------------------------
# 2. Сравнение моделей по информационным критериям
# -------------------------------
for name, res in results_ar_garch.items():
    print(name, "AIC:", res.aic, "BIC:", res.bic)

# -------------------------------
# 3. Кросс-валидация моделей (rolling forecast)
# -------------------------------
from sklearn.metrics import mean_squared_error

rolling_window = 200  # пример: последние 200 дней для обучения
forecast_horizon = 1
mses = {}

for name, m in zip(results_ar_garch.keys(), ar_garch_models):
    errors = []
    for i in range(rolling_window, len(y)-forecast_horizon):
        train = y[i-rolling_window:i]
        model = arch_model(train, mean='AR', lags=m["ar"], 
                           vol='GARCH', p=m["p"], o=m["o"], q=m["q"], power=m["lambda"])
        fit = model.fit(disp='off')
        pred = fit.forecast(horizon=forecast_horizon).mean.iloc[-1,0]
        errors.append((y[i] - pred)**2)
    mses[name] = np.mean(errors)

best_model_name = min(mses, key=mses.get)
print("Лучший вариант по кросс-валидации:", best_model_name)

# -------------------------------
# 4. Подгонка AR-EGARCH и AR-APARCH моделей
# -------------------------------
egarch_aparch_models = [
    {"vol":"EGARCH", "p":1, "o":0, "q":1, "ar":1},
    {"vol":"EGARCH", "p":1, "o":1, "q":1, "ar":1},
    {"vol":"APARCH", "p":1, "o":0, "q":1, "ar":1},
    {"vol":"APARCH", "p":1, "o":1, "q":1, "ar":1},
]

results_egarch_aparch = {}
for m in egarch_aparch_models:
    model_name = f'AR({m["ar"]})-{m["vol"]}({m["p"]},{m["o"]},{m["q"]})'
    am = arch_model(y, mean='AR', lags=m["ar"], vol=m["vol"], p=m["p"], o=m["o"], q=m["q"])
    res = am.fit(disp='off')
    results_egarch_aparch[model_name] = res
    print(res.summary())
    
    forecast = res.forecast(horizon=10)
    print(f"Прогноз волатильности для {model_name}:\n", forecast.variance[-1:])

# Сравнение по AIC/BIC
for name, res in results_egarch_aparch.items():
    print(name, "AIC:", res.aic, "BIC:", res.bic)

```


Задание 3

<img width="828" height="690" alt="image" src="https://github.com/user-attachments/assets/df9b7419-0746-4c3f-8660-1fc12ae572eb" />

Текст

```
\begin{exercise}
Пусть ряд \(y_t\) -- первая разность 10-летней ставки (treasury securities  with constant maturity, \textbf{дневные данные}) с 2010 по н.в.
\begin{enumerate}
	\item Подгоните модели AR-GARCH(p,o,q)
	\begin{center}
	\begin{tabular}{l|c}
		Модель & \(\lambda\) \\ \hline
		AR(1)-GARCH(1,0,1) & 2 \\
		AR(1)-GARCH(1,0,1) & 1 \\
		AR(2)-GARCH(1,0,1) & 2 \\
		AR(2)-GARCH(1,0,1) & 1 \\ \hline
	\end{tabular}
	\end{center} 
	и постройте прогноз на 10 периодов для ряда и его волатильности.
	\item Сравните модели по информационным критериям. Какая предпочтительней?
	\item Проведите кросс-валидацию моделей. Какая предпочтительней?
	\item Подгоните модели
	\begin{center}
		\begin{tabular}{l}
			Модель  \\ \hline
			AR(1)-EGARCH(1,0,1) \\
			AR(1)-EGARCH(1,1,1) \\
			AR(1)-APARCH(1,0,1) \\
			AR(1)-APARCH(1,1,1)
		\end{tabular}
	\end{center} 
	и постройте прогноз на 10 периодов для ряда и его волатильности.
	\item Какая предпочтительней?
\end{enumerate}
\end{exercise}
```

Ответ

```
# -------------------------------
# Импорт библиотек
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# -------------------------------
# Предположим, y — это первая разность 10-летней ставки (дневные данные)
# y = pd.Series(..., index=pd.date_range(...))
# -------------------------------

# -------------------------------
# 1. Подгонка AR-GARCH(p,o,q) моделей
# -------------------------------
ar_garch_models = [
    {"ar":1, "p":1, "o":0, "q":1, "lambda":2},
    {"ar":1, "p":1, "o":0, "q":1, "lambda":1},
    {"ar":2, "p":1, "o":0, "q":1, "lambda":2},
    {"ar":2, "p":1, "o":0, "q":1, "lambda":1},
]

results_ar_garch = {}
for m in ar_garch_models:
    model_name = f'AR({m["ar"]})-GARCH({m["p"]},{m["o"]},{m["q"]})_lambda{m["lambda"]}'
    am = arch_model(y, mean='AR', lags=m["ar"], 
                    vol='GARCH', p=m["p"], o=m["o"], q=m["q"], power=m["lambda"],
                    dist='normal')
    res = am.fit(disp='off')
    results_ar_garch[model_name] = res
    print(res.summary())
    
    # Прогноз на 10 периодов вперед
    forecast = res.forecast(horizon=10)
    print(f"Прогноз волатильности для {model_name}:\n", forecast.variance[-1:])

# -------------------------------
# 2. Сравнение моделей по информационным критериям
# -------------------------------
for name, res in results_ar_garch.items():
    print(name, "AIC:", res.aic, "BIC:", res.bic)

# -------------------------------
# 3. Кросс-валидация моделей (rolling forecast)
# -------------------------------
from sklearn.metrics import mean_squared_error

rolling_window = 200  # пример: последние 200 дней для обучения
forecast_horizon = 1
mses = {}

for name, m in zip(results_ar_garch.keys(), ar_garch_models):
    errors = []
    for i in range(rolling_window, len(y)-forecast_horizon):
        train = y[i-rolling_window:i]
        model = arch_model(train, mean='AR', lags=m["ar"], 
                           vol='GARCH', p=m["p"], o=m["o"], q=m["q"], power=m["lambda"])
        fit = model.fit(disp='off')
        pred = fit.forecast(horizon=forecast_horizon).mean.iloc[-1,0]
        errors.append((y[i] - pred)**2)
    mses[name] = np.mean(errors)

best_model_name = min(mses, key=mses.get)
print("Лучший вариант по кросс-валидации:", best_model_name)

# -------------------------------
# 4. Подгонка AR-EGARCH и AR-APARCH моделей
# -------------------------------
egarch_aparch_models = [
    {"vol":"EGARCH", "p":1, "o":0, "q":1, "ar":1},
    {"vol":"EGARCH", "p":1, "o":1, "q":1, "ar":1},
    {"vol":"APARCH", "p":1, "o":0, "q":1, "ar":1},
    {"vol":"APARCH", "p":1, "o":1, "q":1, "ar":1},
]

results_egarch_aparch = {}
for m in egarch_aparch_models:
    model_name = f'AR({m["ar"]})-{m["vol"]}({m["p"]},{m["o"]},{m["q"]})'
    am = arch_model(y, mean='AR', lags=m["ar"], vol=m["vol"], p=m["p"], o=m["o"], q=m["q"])
    res = am.fit(disp='off')
    results_egarch_aparch[model_name] = res
    print(res.summary())
    
    forecast = res.forecast(horizon=10)
    print(f"Прогноз волатильности для {model_name}:\n", forecast.variance[-1:])

# Сравнение по AIC/BIC
for name, res in results_egarch_aparch.items():
    print(name, "AIC:", res.aic, "BIC:", res.bic)

```

Задание 4

<img width="757" height="1097" alt="image" src="https://github.com/user-attachments/assets/a87f0b66-014c-454a-af6c-03b81663a55f" />


Текст

```
\begin{exercise}
Пусть ряд \(y_t\) -- лог-доходность S\&P500 (\textbf{дневные данные}) с 2010 по н.в.
\begin{enumerate}
	\item Подгоните модели
	\begin{center}
	\begin{tabular}{l|c}
		Модель & \(\lambda\) \\ \hline
		AR(1)-GARCH(1,0,1) & 2 \\
		AR(1)-GARCH(1,0,1) & 1 \\
		AR(1)-GARCH(1,1,1) & 2 \\
		AR(1)-GARCH(1,1,1) & 1 \\ \hline
	\end{tabular}
	\end{center} 
	и постройте прогноз на 10 периодов для ряда и его волатильности. 
	\item Сравните модели по информационным критериям. Какая предпочтительней?
	\item Проведите кросс-валидацию моделей. Какая предпочтительней?
	\item Подгоните модели
	\begin{center}
		\begin{tabular}{l}
			Модель  \\ \hline
			AR(1)-EGARCH(1,0,1) \\
			AR(1)-EGARCH(1,1,1) \\
			AR(1)-APARCH(1,0,1) \\
			AR(1)-APARCH(1,1,1)
		\end{tabular}
	\end{center} 
	и постройте прогноз на 10 периодов для ряда и его волатильности.
	\item Какая предпочтительней?
\end{enumerate}
\end{exercise}
```

Ответ 

```
# -------------------------------
# Импорт библиотек
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error

# -------------------------------
# Пусть y — это лог-доходность S&P500 (дневные данные)
# y = pd.Series(..., index=pd.date_range(...))
# -------------------------------

# -------------------------------
# 1. Подгонка AR-GARCH(p,o,q) моделей
# -------------------------------
ar_garch_models = [
    {"ar":1, "p":1, "o":0, "q":1, "lambda":2},
    {"ar":1, "p":1, "o":0, "q":1, "lambda":1},
    {"ar":1, "p":1, "o":1, "q":1, "lambda":2},
    {"ar":1, "p":1, "o":1, "q":1, "lambda":1},
]

results_ar_garch = {}
for m in ar_garch_models:
    model_name = f'AR({m["ar"]})-GARCH({m["p"]},{m["o"]},{m["q"]})_lambda{m["lambda"]}'
    am = arch_model(y, mean='AR', lags=m["ar"], 
                    vol='GARCH', p=m["p"], o=m["o"], q=m["q"], power=m["lambda"],
                    dist='normal')
    res = am.fit(disp='off')
    results_ar_garch[model_name] = res
    print(res.summary())
    
    # Прогноз на 10 периодов вперед
    forecast = res.forecast(horizon=10)
    print(f"Прогноз волатильности для {model_name}:\n", forecast.variance[-1:])

# -------------------------------
# 2. Сравнение моделей по информационным критериям
# -------------------------------
for name, res in results_ar_garch.items():
    print(name, "AIC:", res.aic, "BIC:", res.bic)

# -------------------------------
# 3. Кросс-валидация моделей (rolling forecast)
# -------------------------------
rolling_window = 200  # последние 200 дней для обучения
forecast_horizon = 1
mses = {}

for name, m in zip(results_ar_garch.keys(), ar_garch_models):
    errors = []
    for i in range(rolling_window, len(y)-forecast_horizon):
        train = y[i-rolling_window:i]
        model = arch_model(train, mean='AR', lags=m["ar"], 
                           vol='GARCH', p=m["p"], o=m["o"], q=m["q"], power=m["lambda"])
        fit = model.fit(disp='off')
        pred = fit.forecast(horizon=forecast_horizon).mean.iloc[-1,0]
        errors.append((y[i] - pred)**2)
    mses[name] = np.mean(errors)

best_model_name = min(mses, key=mses.get)
print("Лучший вариант по кросс-валидации:", best_model_name)

# -------------------------------
# 4. Подгонка AR-EGARCH и AR-APARCH моделей
# -------------------------------
egarch_aparch_models = [
    {"vol":"EGARCH", "p":1, "o":0, "q":1, "ar":1},
    {"vol":"EGARCH", "p":1, "o":1, "q":1, "ar":1},
    {"vol":"APARCH", "p":1, "o":0, "q":1, "ar":1},
    {"vol":"APARCH", "p":1, "o":1, "q":1, "ar":1},
]

results_egarch_aparch = {}
for m in egarch_aparch_models:
    model_name = f'AR({m["ar"]})-{m["vol"]}({m["p"]},{m["o"]},{m["q"]})'
    am = arch_model(y, mean='AR', lags=m["ar"], vol=m["vol"], p=m["p"], o=m["o"], q=m["q"])
    res = am.fit(disp='off')
    results_egarch_aparch[model_name] = res
    print(res.summary())
    
    forecast = res.forecast(horizon=10)
    print(f"Прогноз волатильности для {model_name}:\n", forecast.variance[-1:])

# Сравнение по AIC/BIC
for name, res in results_egarch_aparch.items():
    print(name, "AIC:", res.aic, "BIC:", res.bic)

```


5 Многомерные ряды. Модель VAR/VECM. Коинтеграция

Задание 1

<img width="804" height="1058" alt="image" src="https://github.com/user-attachments/assets/1d13ba69-78de-486b-9535-d2e2dca70d3b" />

Текст

```
\begin{exercise}
Рассмотрим \textbf{недельные} данные с 2000 г по н.в. по следующим переменными
\begin{itemize}
	\item первая разность 3-месячной ставки (3-Month Treasury Bill)
	\item первая разность 6-месячной ставки (6-Month Treasury Bill)
	\item первая разность 10-летней ставки (Treasury Securities at 10-Year Constant Maturity)
\end{itemize}
Сформируйте многомерный ряд и визуализируйте его.
\begin{enumerate}
	\item Фиксированный порядок
	\begin{enumerate}
		\item Подгоните модели VAR(1), VAR(2), VAR(3)
		\item постройте прогноз на 10 периодов по каждой модели
		\item Проведите кросс-валидацию моделей. Какая предпочтительней?
	\end{enumerate}
	\item <<Оптимизация>> порядка
	\begin{enumerate}
		\item Подгоните модель VAR <<оптимального>> порядка
		\item Проведите её диагностику
		\item Постройте прогноз на 5 периодов. Постройте FEVD
		\item Постройте IRF, использую исходное упорядочивание переменных
		\item Проведите тест Гренджера на причинность
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# -------------------------------
# 1. Импорт библиотек
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# -------------------------------
# 2. Загрузка и подготовка данных
# -------------------------------
# Предположим, что у вас есть CSV с колонками:
# 'date', 'diff_3m', 'diff_6m', 'diff_10y'
df = pd.read_csv('weekly_rates.csv', parse_dates=['date'], index_col='date')

# Формируем многомерный ряд
y = df[['diff_3m', 'diff_6m', 'diff_10y']]

# Визуализация
y.plot(figsize=(12,6), title="Первая разность процентных ставок")
plt.show()

# -------------------------------
# 3. VAR фиксированного порядка
# -------------------------------
orders = [1,2,3]
var_models = {}
for p in orders:
    model = VAR(y)
    res = model.fit(p)
    var_models[f'VAR({p})'] = res
    print(f"VAR({p}) summary:\n", res.summary())
    
    # Прогноз на 10 периодов
    forecast = res.forecast(y.values[-p:], steps=10)
    forecast_df = pd.DataFrame(forecast, columns=y.columns)
    print(f"Прогноз VAR({p}):\n", forecast_df)

# -------------------------------
# 4. Кросс-валидация моделей VAR
# -------------------------------
# Rolling forecast
rolling_window = 200
forecast_horizon = 1
mses = {}

for name, res in var_models.items():
    p = res.k_ar
    errors = []
    for i in range(rolling_window, len(y)-forecast_horizon):
        train = y.iloc[i-rolling_window:i]
        model = VAR(train)
        fit = model.fit(p)
        pred = fit.forecast(train.values[-p:], steps=forecast_horizon)[0]
        errors.append(np.mean((y.iloc[i].values - pred)**2))
    mses[name] = np.mean(errors)

best_var_fixed = min(mses, key=mses.get)
print("Лучший фиксированный VAR по MSE:", best_var_fixed)

# -------------------------------
# 5. Оптимизация порядка VAR
# -------------------------------
model_opt = VAR(y)
res_opt = model_opt.select_order(maxlags=10)
print("Рекомендованный порядок по критериям:\n", res_opt.summary())

# Подгонка VAR оптимального порядка
p_opt = res_opt.selected_orders['aic']  # например, AIC
var_opt = model_opt.fit(p_opt)
print(var_opt.summary())

# Прогноз на 5 периодов
forecast_opt = var_opt.forecast(y.values[-p_opt:], steps=5)
forecast_opt_df = pd.DataFrame(forecast_opt, columns=y.columns)
print("Прогноз VAR оптимального порядка:\n", forecast_opt_df)

# FEVD (Forecast Error Variance Decomposition)
fevd = var_opt.fevd(5)
fevd.summary()  # покажет вклад шока каждой переменной

# IRF (Impulse Response Function)
irf = var_opt.irf(10)
irf.plot(orth=False)  # исходное упорядочивание переменных
plt.show()

# -------------------------------
# 6. Тест Гренджера на причинность
# -------------------------------
# Например, проверка, вызывает ли diff_3m diff_6m
granger_test = grangercausalitytests(y[['diff_6m','diff_3m']], maxlag=5, verbose=True)

```

Задание 2

<img width="732" height="634" alt="image" src="https://github.com/user-attachments/assets/59b805c2-6a45-4ae3-96d0-8c847a262eaf" />


Текст

```
\begin{exercise}
Рассмотрим \textbf{дневные} данные с 2000 г по н.в. по следующим переменными
\begin{itemize}
	\item первая разность 3-месячной ставки (3-Month Treasury Bill)
	\item первая разность 6-месячной ставки (6-Month Treasury Bill)
	\item первая разность 10-летней ставки (Treasury Securities at 10-Year Constant Maturity)
\end{itemize}
Сформируйте многомерный ряд и визуализируйте его.
\begin{enumerate}
	\item Фиксированный порядок
	\begin{enumerate}
		\item Подгоните модели VAR(1), VAR(2), VAR(3)
		\item постройте прогноз на 10 периодов по каждой модели
		\item Проведите кросс-валидацию моделей. Какая предпочтительней?
	\end{enumerate}
	\item <<Оптимизация>> порядка
	\begin{enumerate}
		\item Подгоните модель VAR <<оптимального>> порядка
		\item Проведите её диагностику
		\item Постройте прогноз на 5 периодов. Постройте FEVD
		\item Постройте IRF, использую исходное упорядочивание переменных
		\item Проведите тест Гренджера на причинность
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответы

```
# -------------------------------
# 1. Импорт библиотек
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# -------------------------------
# 2. Загрузка и подготовка данных
# -------------------------------
# Предположим, что у вас есть CSV с колонками:
# 'date', 'diff_3m', 'diff_6m', 'diff_10y'
df = pd.read_csv('daily_rates.csv', parse_dates=['date'], index_col='date')

# Формируем многомерный ряд
y = df[['diff_3m', 'diff_6m', 'diff_10y']]

# Визуализация
y.plot(figsize=(12,6), title="Первая разность процентных ставок")
plt.show()

# -------------------------------
# 3. VAR фиксированного порядка
# -------------------------------
orders = [1,2,3]
var_models = {}
for p in orders:
    model = VAR(y)
    res = model.fit(p)
    var_models[f'VAR({p})'] = res
    print(f"VAR({p}) summary:\n", res.summary())
    
    # Прогноз на 10 периодов
    forecast = res.forecast(y.values[-p:], steps=10)
    forecast_df = pd.DataFrame(forecast, columns=y.columns)
    print(f"Прогноз VAR({p}):\n", forecast_df)

# -------------------------------
# 4. Кросс-валидация моделей VAR
# -------------------------------
rolling_window = 250  # например, 1 год торговых дней
forecast_horizon = 1
mses = {}

for name, res in var_models.items():
    p = res.k_ar
    errors = []
    for i in range(rolling_window, len(y)-forecast_horizon):
        train = y.iloc[i-rolling_window:i]
        model = VAR(train)
        fit = model.fit(p)
        pred = fit.forecast(train.values[-p:], steps=forecast_horizon)[0]
        errors.append(np.mean((y.iloc[i].values - pred)**2))
    mses[name] = np.mean(errors)

best_var_fixed = min(mses, key=mses.get)
print("Лучший фиксированный VAR по MSE:", best_var_fixed)

# -------------------------------
# 5. Оптимизация порядка VAR
# -------------------------------
model_opt = VAR(y)
res_opt = model_opt.select_order(maxlags=15)
print("Рекомендованный порядок по критериям:\n", res_opt.summary())

# Подгонка VAR оптимального порядка
p_opt = res_opt.selected_orders['aic']  # например, по AIC
var_opt = model_opt.fit(p_opt)
print(var_opt.summary())

# Прогноз на 5 периодов
forecast_opt = var_opt.forecast(y.values[-p_opt:], steps=5)
forecast_opt_df = pd.DataFrame(forecast_opt, columns=y.columns)
print("Прогноз VAR оптимального порядка:\n", forecast_opt_df)

# -------------------------------
# 6. FEVD (Forecast Error Variance Decomposition)
# -------------------------------
fevd = var_opt.fevd(5)
fevd.summary()  # покажет вклад шока каждой переменной

# -------------------------------
# 7. IRF (Impulse Response Function)
# -------------------------------
irf = var_opt.irf(10)
irf.plot(orth=False)  # исходное упорядочивание переменных
plt.show()

# -------------------------------
# 8. Тест Гренджера на причинность
# -------------------------------
# Пример: проверяем, вызывает ли diff_3m diff_6m
granger_test = grangercausalitytests(y[['diff_6m','diff_3m']], maxlag=5, verbose=True)

```

Задание 3

<img width="761" height="1105" alt="image" src="https://github.com/user-attachments/assets/c43f0db3-bac7-4a45-ba2f-fbe7d3f3c233" />


Текст

```
\begin{exercise}
Рассмотрим \textbf{месячные} данные с 1995 г по н.в. по следующим переменными
\begin{itemize}
	\item первая разность 3-месячной ставки (3-Month Treasury Bill)
	\item первая разность 6-месячной ставки (6-Month Treasury Bill)
	\item первая разность 10-летней ставки (Treasury Securities at 10-Year Constant Maturity)
	\item лог-доходность US M2
\end{itemize}
Сформируйте многомерный ряд и визуализируйте его.
\begin{enumerate}
	\item Фиксированный порядок
	\begin{enumerate}
		\item Подгоните модели VAR(1), VAR(2), VAR(3)
		\item постройте прогноз на 10 периодов по каждой модели
		\item Проведите кросс-валидацию моделей. Какая предпочтительней?
	\end{enumerate}
	\item <<Оптимизация>> порядка
	\begin{enumerate}
		\item Подгоните модель VAR <<оптимального>> порядка
		\item Проведите её диагностику
		\item Постройте прогноз на 5 периодов. Постройте FEVD
		\item Постройте IRF, использую исходное упорядочивание переменных
		\item Проведите тест Гренджера на причинность
	\end{enumerate}
\end{enumerate}
\end{exercise}
```

Ответ

```
# -------------------------------
# 1. Импорт библиотек
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# -------------------------------
# 2. Загрузка данных
# -------------------------------
# CSV должен содержать колонки:
# 'date', 'diff_3m', 'diff_6m', 'diff_10y', 'log_ret_m2'
df = pd.read_csv('monthly_rates_m2.csv', parse_dates=['date'], index_col='date')

# Формируем многомерный ряд
y = df[['diff_3m', 'diff_6m', 'diff_10y', 'log_ret_m2']]

# Визуализация
y.plot(figsize=(12,6), title="Месячные изменения ставок и M2")
plt.show()

# -------------------------------
# 3. VAR фиксированного порядка
# -------------------------------
orders = [1,2,3]
var_models = {}
for p in orders:
    model = VAR(y)
    res = model.fit(p)
    var_models[f'VAR({p})'] = res
    print(f"VAR({p}) summary:\n", res.summary())
    
    # Прогноз на 10 месяцев
    forecast = res.forecast(y.values[-p:], steps=10)
    forecast_df = pd.DataFrame(forecast, columns=y.columns)
    print(f"Прогноз VAR({p}):\n", forecast_df)

# -------------------------------
# 4. Кросс-валидация моделей VAR
# -------------------------------
rolling_window = 60  # 5 лет месяцев
forecast_horizon = 1
mses = {}

for name, res in var_models.items():
    p = res.k_ar
    errors = []
    for i in range(rolling_window, len(y)-forecast_horizon):
        train = y.iloc[i-rolling_window:i]
        model = VAR(train)
        fit = model.fit(p)
        pred = fit.forecast(train.values[-p:], steps=forecast_horizon)[0]
        errors.append(np.mean((y.iloc[i].values - pred)**2))
    mses[name] = np.mean(errors)

best_var_fixed = min(mses, key=mses.get)
print("Лучший фиксированный VAR по MSE:", best_var_fixed)

# -------------------------------
# 5. Оптимизация порядка VAR
# -------------------------------
model_opt = VAR(y)
res_opt = model_opt.select_order(maxlags=12)  # макс 1 год
print("Рекомендованный порядок по критериям:\n", res_opt.summary())

p_opt = res_opt.selected_orders['aic']  # например, по AIC
var_opt = model_opt.fit(p_opt)
print(var_opt.summary())

# Прогноз на 5 месяцев
forecast_opt = var_opt.forecast(y.values[-p_opt:], steps=5)
forecast_opt_df = pd.DataFrame(forecast_opt, columns=y.columns)
print("Прогноз VAR оптимального порядка:\n", forecast_opt_df)

# -------------------------------
# 6. FEVD
# -------------------------------
fevd = var_opt.fevd(5)
fevd.summary()  # вклад шока каждой переменной

# -------------------------------
# 7. IRF
# -------------------------------
irf = var_opt.irf(10)
irf.plot(orth=False)
plt.show()

# -------------------------------
# 8. Тест Гренджера на причинность
# -------------------------------
# Пример: проверяем, вызывает ли diff_3m diff_6m
granger_test = grangercausalitytests(y[['diff_6m','diff_3m']], maxlag=5, verbose=True)

```

Задание 4

<img width="804" height="913" alt="image" src="https://github.com/user-attachments/assets/b2068997-3803-4a27-a643-59b200423ed0" />

Текст

```
\begin{exercise}[VECM]
Рассмотрим \textbf{недельные} данные с 2005 г по н.в. по следующим переменными
\begin{itemize}
	\item 3-месячная ставки (3-Month Treasury Bill)
	\item 6-месячная ставки (6-Month Treasury Bill)
	\item 1-летняя ставка (Treasury Securities at 1-Year Constant Maturity)
	\item 10-летняя ставка (Treasury Securities at 10-Year Constant Maturity)
\end{itemize}
Сформируйте многомерный ряд и визуализируйте его.
\begin{enumerate}
	\item Найдите ранг коинтеграции
	\item Оцените модель VECM <<оптимального>> порядка
	\item Проведите её диагностику
	\item Постройте прогноз на 5 периодов. Постройте FEVD
	\item Постройте IRF, использую исходное упорядочивание переменных
	\item Проведите тест Гренджера на причинность
\end{enumerate}
\end{exercise}
```

Ответ

```
# -------------------------------
# 1. Импорт библиотек
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.stattools import grangercausalitytests

# -------------------------------
# 2. Загрузка данных
# -------------------------------
# CSV должен содержать колонки:
# 'date', 'rate_3m', 'rate_6m', 'rate_1y', 'rate_10y'
df = pd.read_csv('weekly_rates.csv', parse_dates=['date'], index_col='date')

# Формируем многомерный ряд
y = df[['rate_3m','rate_6m','rate_1y','rate_10y']]

# Визуализация
y.plot(figsize=(12,6), title="Недельные ставки")
plt.show()

# -------------------------------
# 3. Ранг коинтеграции (Johansen)
# -------------------------------
jres = coint_johansen(y, det_order=0, k_ar_diff=5)  # k_ar_diff = кол-во лагов в Δy
print("Eigenvalues:", jres.lr1)
print("Critical values (90/95/99%):\n", jres.cvt)

# Ранг коинтеграции можно определить по сравнению lr1 с критическими значениями
# Например:
for i, val in enumerate(jres.lr1):
    print(f"r ≤ {i}? LR1={val}, 95% критическое={jres.cvt[i,1]}")

# -------------------------------
# 4. Подгонка VECM <<оптимального>> порядка
# -------------------------------
# Определение оптимального числа лагов для Δy
order_res = select_order(y, maxlags=10, deterministic="co")
print("Рекомендованный порядок Δy:", order_res.summary())

p_opt = order_res.aic  # например, по AIC

vecm = VECM(y, k_ar_diff=p_opt, coint_rank=1, deterministic="co")  # r=1, можно менять
vecm_res = vecm.fit()
print(vecm_res.summary())

# -------------------------------
# 5. Прогноз на 5 периодов
# -------------------------------
forecast = vecm_res.predict(steps=5)
forecast_df = pd.DataFrame(forecast, columns=y.columns)
print("Прогноз VECM на 5 периодов:\n", forecast_df)

# -------------------------------
# 6. FEVD (Forecast Error Variance Decomposition)
# -------------------------------
# Для FEVD используем встроенные функции VAR после преобразования VECM в VAR
from statsmodels.tsa.vector_ar.var_model import VAR

vecm_to_var = vecm_res.predict(steps=len(y))  # только для построения VAR
var_model = VAR(y.diff().dropna())
var_res = var_model.fit(p_opt)
fevd = var_res.fevd(5)
fevd.summary()

# -------------------------------
# 7. IRF (Impulse Response Function)
# -------------------------------
irf = var_res.irf(10)
irf.plot(orth=False)
plt.show()

# -------------------------------
# 8. Тест Гренджера на причинность
# -------------------------------
# Пример: проверяем, вызывает ли 3M 6M
granger_test = grangercausalitytests(y[['rate_6m','rate_3m']], maxlag=5, verbose=True)

```

Задание 5


<img width="776" height="509" alt="image" src="https://github.com/user-attachments/assets/c4bcea33-9c44-4d49-9449-dc5144a24d71" />

Текст

```
\begin{exercise}[VECM]
Рассмотрим \textbf{месячные} данные с 2005 г по н.в. по следующим переменными
\begin{itemize}
	\item 3-месячная ставки (3-Month Treasury Bill)
	\item 6-месячная ставки (6-Month Treasury Bill)
	\item 1-летняя ставка (Treasury Securities at 1-Year Constant Maturity)
	\item 10-летняя ставка (Treasury Securities at 10-Year Constant Maturity)
\end{itemize}
Сформируйте многомерный ряд и визуализируйте его.
\begin{enumerate}
	\item Найдите ранг коинтеграции
	\item Оцените модель VECM <<оптимального>> порядка
	\item Проведите её диагностику
	\item Постройте прогноз на 5 периодов. Постройте FEVD
	\item Постройте IRF, использую исходное упорядочивание переменных
	\item Проведите тест Гренджера на причинность
\end{enumerate}
\end{exercise}
```

Ответ

```
# -------------------------------
# 1. Импорт библиотек
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

# -------------------------------
# 2. Загрузка данных
# -------------------------------
# CSV должен содержать колонки:
# 'date', 'rate_3m', 'rate_6m', 'rate_1y', 'rate_10y'
df = pd.read_csv('monthly_rates.csv', parse_dates=['date'], index_col='date')

# Формируем многомерный ряд
y = df[['rate_3m','rate_6m','rate_1y','rate_10y']]

# Визуализация
y.plot(figsize=(12,6), title="Месячные ставки")
plt.show()

# -------------------------------
# 3. Ранг коинтеграции (Johansen)
# -------------------------------
jres = coint_johansen(y, det_order=0, k_ar_diff=12)  # 12 лагов в Δy (пример)
print("Eigenvalues:", jres.lr1)
print("Critical values (90/95/99%):\n", jres.cvt)

# Определение r по сравнению lr1 с критическими значениями
for i, val in enumerate(jres.lr1):
    print(f"r ≤ {i}? LR1={val}, 95% критическое={jres.cvt[i,1]}")

# -------------------------------
# 4. Подгонка VECM <<оптимального>> порядка
# -------------------------------
order_res = select_order(y, maxlags=10, deterministic="co")
print("Рекомендованный порядок Δy:", order_res.summary())

p_opt = order_res.aic  # оптимальный лаг по AIC

vecm = VECM(y, k_ar_diff=p_opt, coint_rank=1, deterministic="co")  # r=1, можно менять
vecm_res = vecm.fit()
print(vecm_res.summary())

# -------------------------------
# 5. Прогноз на 5 периодов
# -------------------------------
forecast = vecm_res.predict(steps=5)
forecast_df = pd.DataFrame(forecast, columns=y.columns)
print("Прогноз VECM на 5 периодов:\n", forecast_df)

# -------------------------------
# 6. FEVD (Forecast Error Variance Decomposition)
# -------------------------------
var_model = VAR(y.diff().dropna())
var_res = var_model.fit(p_opt)
fevd = var_res.fevd(5)
fevd.summary()

# -------------------------------
# 7. IRF (Impulse Response Function)
# -------------------------------
irf = var_res.irf(10)
irf.plot(orth=False)
plt.show()

# -------------------------------
# 8. Тест Гренджера на причинность
# -------------------------------
# Пример: проверяем, вызывает ли 3M 6M
granger_test = grangercausalitytests(y[['rate_6m','rate_3m']], maxlag=5, verbose=True)

```


Задание 6

<img width="762" height="960" alt="image" src="https://github.com/user-attachments/assets/80a740b6-7e41-472d-897c-6253764aef41" />


Текст

```
\begin{exercise}[VECM]
Рассмотрим \textbf{месячные} данные с 1995 г по н.в. по следующим переменными
\begin{itemize}
	\item 3-месячная ставки (3-Month Treasury Bill)
	\item 6-месячная ставки (6-Month Treasury Bill)
	\item 1-летняя ставка (Treasury Securities at 1-Year Constant Maturity)
	\item 10-летняя ставка (Treasury Securities at 10-Year Constant Maturity)
	\item лог-M2
\end{itemize}
Сформируйте многомерный ряд и визуализируйте его.
\begin{enumerate}
	\item Найдите ранг коинтеграции
	\item Оцените модель VECM <<оптимального>> порядка
	\item Проведите её диагностику
	\item Постройте прогноз на 5 периодов. Постройте FEVD
	\item Постройте IRF, использую исходное упорядочивание переменных
	\item Проведите тест Гренджера на причинность
\end{enumerate}
\end{exercise}
```

Ответ

```
# -------------------------------
# 1. Импорт библиотек
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# -------------------------------
# 2. Загрузка данных
# -------------------------------
# CSV должен содержать колонки:
# 'date','rate_3m','rate_6m','rate_1y','rate_10y','log_m2'
df = pd.read_csv('monthly_data.csv', parse_dates=['date'], index_col='date')

# Формируем многомерный ряд
y = df[['rate_3m','rate_6m','rate_1y','rate_10y','log_m2']]

# Визуализация
y.plot(figsize=(12,6), title="Месячные ставки и log-M2")
plt.show()

# -------------------------------
# 3. Ранг коинтеграции (Johansen)
# -------------------------------
jres = coint_johansen(y, det_order=0, k_ar_diff=12)  # 12 лагов в Δy (пример)
print("Eigenvalues:", jres.lr1)
print("Critical values (90/95/99%):\n", jres.cvt)

# Определение r по сравнению lr1 с критическими значениями
for i, val in enumerate(jres.lr1):
    print(f"r ≤ {i}? LR1={val}, 95% критическое={jres.cvt[i,1]}")

# -------------------------------
# 4. Подгонка VECM <<оптимального>> порядка
# -------------------------------
order_res = select_order(y, maxlags=12, deterministic="co")
print("Рекомендованный лаг Δy:", order_res.summary())

p_opt = order_res.aic  # используем оптимальный лаг по AIC
vecm = VECM(y, k_ar_diff=p_opt, coint_rank=1, deterministic="co")  # r=1, можно менять
vecm_res = vecm.fit()
print(vecm_res.summary())

# -------------------------------
# 5. Прогноз на 5 периодов
# -------------------------------
forecast = vecm_res.predict(steps=5)
forecast_df = pd.DataFrame(forecast, columns=y.columns)
print("Прогноз VECM на 5 периодов:\n", forecast_df)

# -------------------------------
# 6. FEVD (Forecast Error Variance Decomposition)
# -------------------------------
var_model = VAR(y.diff().dropna())
var_res = var_model.fit(p_opt)
fevd = var_res.fevd(5)
fevd.summary()

# -------------------------------
# 7. IRF (Impulse Response Function)
# -------------------------------
irf = var_res.irf(10)
irf.plot(orth=False)
plt.show()

# -------------------------------
# 8. Тест Гренджера на причинность
# -------------------------------
# Пример: проверяем, вызывает ли 3M 6M
granger_test = grangercausalitytests(y[['rate_6m','rate_3m']], maxlag=5, verbose=True)

```


Задание 7

<img width="766" height="352" alt="image" src="https://github.com/user-attachments/assets/4363ab20-0666-4f83-afe2-60924a6ff5d5" />

Текст

```
\begin{exercise}
Рассмотрим VAR(1)
\begin{align*}
	\vectx_t&=\matrixA\vectx_{t-1}+\vectu_t &
	\vectx_t&=\begin{pmatrix} x_t \\ y_t \end{pmatrix} &
	\vectu_t&=\begin{pmatrix} u_t \\ v_t \end{pmatrix}
\end{align*}
где 
\begin{align*}
	\vectu_t&\sim WN(0,\Sigma) &
	\Sigma&=\begin{pmatrix}
	\sigma^2_u & \sigma_{uv} \\ \sigma_{uv} & \sigma^2_v
	\end{pmatrix}>0
\end{align*}
т.е. $u_t\sim WN(0,\sigma_u^2)$,   $v_t\sim WN(0,\sigma_v^2)$,
$\cov(u_t,v_t)=\sigma_{uv}$. 
	
Проверить условие стационарности для следующих матриц
\begin{align*}
	\matrixA&=\begin{pmatrix} 0.5 & 1 \\ 0 & 0.3 \end{pmatrix} &
	&\begin{pmatrix} 0 & 0.5 \\ -0.5 & 0 \end{pmatrix} &
	& \begin{pmatrix} 1 & 3 \\ 0 & 0.2  \end{pmatrix} &
	& \begin{pmatrix} 0 & 1 \\ 0 & 1  \end{pmatrix} &
	& \begin{pmatrix} 1 & 1 \\ 1 & 1  \end{pmatrix} &
	& \begin{pmatrix} 0 & 1 \\ 1 & 1  \end{pmatrix}
\end{align*}
\end{exercise}
```

Ответ

```
import numpy as np

# -------------------------------
# Список матриц A
# -------------------------------
matrices = [
    np.array([[0.5, 1], [0, 0.3]]),
    np.array([[0, 0.5], [-0.5, 0]]),
    np.array([[1, 3], [0, 0.2]]),
    np.array([[0, 1], [0, 1]]),
    np.array([[1, 1], [1, 1]]),
    np.array([[0, 1], [1, 1]])
]

# -------------------------------
# Проверка стационарности
# -------------------------------
for i, A in enumerate(matrices, start=1):
    eigvals = np.linalg.eigvals(A)
    mod_eig = np.abs(eigvals)
    is_stationary = np.all(mod_eig < 1)
    
    print(f"Матрица A{i}:\n{A}")
    print(f"Собственные значения: {eigvals}")
    print(f"Модули: {mod_eig}")
    print(f"Стационарность: {'Да' if is_stationary else 'Нет'}\n")

```

Задание 8

<img width="817" height="831" alt="image" src="https://github.com/user-attachments/assets/7feace0a-e05d-4d27-8447-9292f065bccf" />


Текст

```
\begin{exercise}
Рассмотрим VAR-модели
\begin{align*}
	& \begin{cases} x_t=x_{t-1}+u_t \\ y_t=y_{t-1}+v_t \end{cases} &
	& \begin{cases} x_t=x_{t-1}+u_t \\ y_t=cx_{t}+v_t \end{cases} \\
	& \begin{cases} x_t=x_{t-1}+u_t \\ y_t=x_{t}+x_{t-1}+v_t \end{cases} &
	& \begin{cases} x_t=3x_{t-1}-7y_{t-1}+u_t \\ y_t=x_{t-1}-2.5y_{t-1}+v_t \end{cases}
\end{align*}
\begin{enumerate}
	\item Запишите в матричном виде.
	\item Проверить условие стационарности.
	\item Какие ряды коинтегрированы?
	\begin{itemize}
		\item Если ряды коинтегрированы, то запишите VECM модель
		и найдите коинтеграционные соотношения
		\item Если ряды не коинтегрированы, то запишите VAR-модель для дифференцированных рядов.
	\end{itemize}
\end{enumerate}
\end{exercise}
```

Ответ

```
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Определяем матрицы VAR
# ----------------------------
A_list = [
    np.array([[1, 0], [0, 1]]),     # система 1
    np.array([[1, 0], [1, 0]]),     # система 2 (c=1)
    np.array([[1, 0], [2, 0]]),     # система 3
    np.array([[3, -7], [1, -2.5]])  # система 4
]

# ----------------------------
# 2️⃣ Проверка стационарности
# ----------------------------
print("Стационарность VAR(1) проверка:")
for i, A in enumerate(A_list, start=1):
    eigvals = np.linalg.eigvals(A)
    print(f"\nСистема {i}:")
    print("Матрица A:\n", A)
    print("Собственные значения:", eigvals)
    print("Стационарность:", "Да" if np.all(np.abs(eigvals)<1) else "Нет")

# ----------------------------
# 3️⃣ Генерация искусственных рядов (для примера)
# ----------------------------
# Чтобы показать построение VAR/VECM, сгенерируем случайные данные
np.random.seed(42)
T = 200
data = pd.DataFrame(np.random.randn(T, 2), columns=['x', 'y'])

# ----------------------------
# 4️⃣ Проверка коинтеграции (Johansen test)
# ----------------------------
jres = coint_johansen(data, det_order=0, k_ar_diff=1)
print("\nJohansen test eigenvalues:", jres.lr1)
print("Критические значения 90%, 95%, 99%:", jres.cvt)

# ----------------------------
# 5️⃣ Построение VAR или VECM
# ----------------------------
# Если ряды не стационарны, берем разности
data_diff = data.diff().dropna()

# Пример VAR на разностях
model_var = VAR(data_diff)
results_var = model_var.fit(1)  # VAR(1)
print("\nVAR(1) на разностях:")
print(results_var.summary())

# Пример VECM (если есть коинтеграция)
vecm = VECM(data, k_ar_diff=1, coint_rank=1, deterministic='n')
vecm_res = vecm.fit()
print("\nVECM результат:")
print("Коэффициенты корректирующего механизма (alpha):\n", vecm_res.alpha)
print("Коэффициенты коинтеграции (beta):\n", vecm_res.beta)

# ----------------------------
# 6️⃣ Прогнозирование
# ----------------------------
# VAR прогноз
lag_order = results_var.k_ar
forecast_var = results_var.forecast(data_diff.values[-lag_order:], steps=10)
forecast_var_df = pd.DataFrame(forecast_var, columns=['x','y'])
print("\nVAR прогноз на 10 периодов:\n", forecast_var_df)

# VECM прогноз
vecm_forecast = vecm_res.predict(steps=10)
vecm_forecast_df = pd.DataFrame(vecm_forecast, columns=['x','y'])
print("\nVECM прогноз на 10 периодов:\n", vecm_forecast_df)

# ----------------------------
# 7️⃣ Визуализация
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(data['x'], label='x')
plt.plot(data['y'], label='y')
plt.title("Сгенерированные ряды x и y")
plt.legend()
plt.show()

```


Задание 9

<img width="709" height="268" alt="image" src="https://github.com/user-attachments/assets/bf8f6b86-4474-4481-a498-f6379927d535" />


Текст

```
\begin{exercise}
Рассмотрим $\VAR(1)$
\begin{align*}
	\vectx_t&=\matrixA\vectx_{t-1}+\vectu_t &
	\vectx_t&=\begin{pmatrix} x_t \\ y_t \\ z_t \end{pmatrix} &
	\vectu_t&=\begin{pmatrix} u_t \\ v_t \\ w_t \end{pmatrix}\sim WN(0,\Sigma)
\end{align*}
Проверьте условие стационарности для матриц
\begin{align*}
	\matrixA&=\begin{pmatrix}
	0 & 1 & 3 \\ -1 & 0 & -2 \\ 0 & 0 & 0.5 \end{pmatrix} &
	&\begin{pmatrix}
		0 & 0 & 0.5 \\ 0.5 & 0 & 0 \\0 & 0.5 & 0
	\end{pmatrix}
\end{align*}
\end{exercise}
```

Ответ

```
import numpy as np

# ----------------------------
# 1️⃣ Определяем матрицы A
# ----------------------------
A_list = [
    np.array([[0, 1, 3],
              [-1, 0, -2],
              [0, 0, 0.5]]),
    np.array([[0, 0, 0.5],
              [0.5, 0, 0],
              [0, 0.5, 0]])
]

# ----------------------------
# 2️⃣ Проверка стационарности
# ----------------------------
for i, A in enumerate(A_list, start=1):
    eigvals = np.linalg.eigvals(A)
    print(f"\nМатрица {i}:")
    print(A)
    print("Собственные значения:", eigvals)
    is_stationary = np.all(np.abs(eigvals) < 1)
    print("Стационарна:" , "Да" if is_stationary else "Нет")

```

Задание 10

<img width="775" height="330" alt="image" src="https://github.com/user-attachments/assets/8232fef8-3939-461a-a774-7db744861c87" />


Текст

```
\begin{exercise}
Рассмотрим $\VAR(2)$
\begin{align*}
	\vectx_t&=\matrixA_1\vectx_{t-1}+\matrixA_2\vectx_{t-2}+\vectu_t &
	\vectx_t&=\begin{pmatrix} x_t \\ y_t \end{pmatrix} &
	\vectu_t&=\begin{pmatrix} u_t \\ v_t \end{pmatrix}\sim
	WN(0,\Sigma)
\end{align*}
Проверьте условие стационарности для матриц
\begin{align*}
	1)\,\,\matrixA_1&=\begin{pmatrix} 2 & 3 \\ 0 & 1 \end{pmatrix} &
	\matrixA_2&=\begin{pmatrix} -1 & 3 \\ 0 & -0.25 \end{pmatrix} \\
	2)\,\,\matrixA_1&=\begin{pmatrix} 0 & 0.5 \\ 0.5 & 0 \end{pmatrix} &
	\matrixA_2&=\begin{pmatrix} 0 & -0.25 \\ -0.25 & 0 \end{pmatrix}
\end{align*}
\end{exercise}
```

Ответ

```
import numpy as np

# ----------------------------
# 1️⃣ Определяем матрицы
# ----------------------------
A_matrices = [
    (np.array([[2, 3],
               [0, 1]]),
     np.array([[-1, 3],
               [0, -0.25]])),
    (np.array([[0, 0.5],
               [0.5, 0]]),
     np.array([[0, -0.25],
               [-0.25, 0]]))
]

# ----------------------------
# 2️⃣ Проверка стационарности через companion-матрицу
# ----------------------------
for i, (A1, A2) in enumerate(A_matrices, start=1):
    # размерность
    k = A1.shape[0]
    # создаём companion-матрицу
    top = np.hstack([A1, A2])
    bottom = np.hstack([np.eye(k), np.zeros((k,k))])
    companion = np.vstack([top, bottom])
    
    eigvals = np.linalg.eigvals(companion)
    print(f"\nVAR(2) №{i}")
    print("Companion-матрица:\n", companion)
    print("Собственные значения:", eigvals)
    is_stationary = np.all(np.abs(eigvals) < 1)
    print("Стационарна:" , "Да" if is_stationary else "Нет")

```

Задание 11

<img width="819" height="663" alt="image" src="https://github.com/user-attachments/assets/a873c53f-3f7a-42ba-b5bb-3a34058c2941" />

Текст

```
\begin{exercise}
Рассмотрим модели
\begin{align*}
	& \begin{cases} x_t=2x_{t-1}-x_{t-2}+u_t \\ y_t=1.5y_{t-1}-0.5y_{t-2}+v_t \end{cases} \\
	& \begin{cases} x_t=1.5x_{t-1}+y_{t-1}-0.5x_{t-2}-y_{t-2}+u_t \\
	y_t=-x_{t-1}-0.5y_{t-1}+x_{t-2}+1.5y_{t-2}+v_t \end{cases} \\
	& \begin{cases} x_t=x_{t-1}+u_t \\ y_t=x_{t}+x_{t-1}+v_t \\ z_t= x_{t}+y_{t-1}+w_t \end{cases} \\
	& \begin{cases} x_t=y_{t-1}+u_t \\ y_t=z_{t-1}+v_t \\ z_t= x_{t-1}+w_t \end{cases}
\end{align*}
\begin{enumerate}
	\item Запишите в матричном виде.
	\item Проверить условие стационарности.
	\item Какие ряды коинтегрированы?
	\begin{itemize}
		\item Если ряды коинтегрированы, то запишите VECM модель
		и найдите коинтеграционные соотношения
		\item Если ряды не коинтегрированы, то запишите VAR-модель для дифференцированных рядов.
	\end{itemize}
\end{enumerate}
\end{exercise}
```

Ответ

```
import numpy as np
from numpy.linalg import eigvals

# ----------------------------
# 1️⃣ Определение моделей
# ----------------------------
# Модель 1: x_t=2x_{t-1}-x_{t-2}+u_t, y_t=1.5y_{t-1}-0.5y_{t-2}+v_t
A1_1 = np.array([[2,0], [0,1.5]])
A2_1 = np.array([[-1,0], [0,-0.5]])

# Модель 2
A1_2 = np.array([[1.5,1], [-1,-0.5]])
A2_2 = np.array([[-0.5,-1],[1,1.5]])

# Модель 3: 3 переменные
A1_3 = np.array([[1,0,0], [1,0,0], [1,0,0]])  # верхняя матрица A1
A2_3 = np.array([[0,-1,0],[0,0,0],[0,0,0]])   # верхняя матрица A2 (пример, нужно уточнить)

# Модель 4: 3 переменные
A1_4 = np.array([[0,1,0],[0,0,1],[1,0,0]])
A2_4 = np.zeros((3,3))

models = [
    (A1_1, A2_1, "Model 1"),
    (A1_2, A2_2, "Model 2"),
    (A1_3, A2_3, "Model 3"),
    (A1_4, A2_4, "Model 4")
]

# ----------------------------
# 2️⃣ Функция проверки стационарности VAR(2)
# ----------------------------
def check_stationarity(A1, A2):
    k = A1.shape[0]
    companion = np.vstack([
        np.hstack([A1, A2]),
        np.hstack([np.eye(k), np.zeros((k,k))])
    ])
    eig = eigvals(companion)
    stationary = np.all(np.abs(eig) < 1)
    return eig, stationary

# ----------------------------
# 3️⃣ Проверка моделей
# ----------------------------
for A1, A2, name in models:
    eig, stationary = check_stationarity(A1, A2)
    print(f"\n{name}:")
    print("Собственные значения companion-матрицы:", eig)
    print("Стационарна:" , "Да" if stationary else "Нет")

```
