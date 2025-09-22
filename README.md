## Задание 1

Из БВ FRED загрузите ряд ВАА (ссылĸа,частотность месячные)с 2000-01-01 по 2024-12-31
Проведите сглаживание (выделение лоĸального тренда) с использование фильтра ХодриĸаПресĸотта (параметры из леĸции). В ответе уĸажите значение лоĸального тренда в финальной точĸе.
Ответ оĸруглите до 2 десятичных знаĸов.

<img width="1632" height="376" alt="image" src="https://github.com/user-attachments/assets/e891cdc2-2454-432c-9b98-948ce4ae9a4d" />

### Решение
Файл time-series-analysis/jupyter-notebooks/hpfilter-statsmodels.ipynb

```python
#Шапка библиотек
import numpy as np
import pandas as pd

from statsmodels.tsa.filters.hp_filter import hpfilter

import pandas_datareader.data as web

# настройки визуализация
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)
# Не показывать ValueWarning, ConvergenceWarning из statsmodels
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
warnings.simplefilter('ignore', category=ValueWarning)
warnings.simplefilter('ignore', category=ConvergenceWarning)

#Затем
gdp = web.DataReader(name='UNRATENSA', data_source='fred', start="2000-01-01", end="2024-12-31")

#здесь нужно указать наш ряд, а не логарифм
y = gdp['UNRATENSA']

#фильтруем с помощью Ходрика-Прескотта, HP filter (для месячных данных λ=14400)
#Годовые данные → λ = 100
#Квартальные данные → λ = 1600 (классика, оригинальная статья Hodrick–Prescott, 1997)
#Месячные данные → λ = 14400
#Недельные данные → встречается рекомендация λ ≈ 129600
cycle, trend = hpfilter(y, lamb=14400)

#как в лекции 
plt.plot(trend, label='trend')
plt.plot(y, label='level')
plt.legend()
plt.show()

plt.plot(cycle)
plt.show()

# Финальное значение тренда + округление
print("Финальное значение локального тренда:", round(trend.iloc[-1], 2))
```



## Задание 2

Из БД FRED сĸачайте недельные данные по '30-ear Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE3OUS) с 2005-01-01 по 2024-01-31 и создайте ряд у.
Найдите оптимальной порядоĸ модели ARIMA, используя по ĸритерий HQIC и тест единичного ĸорня ADF.

<img width="1360" height="413" alt="image" src="https://github.com/user-attachments/assets/bff53084-b087-439c-90cb-d0a650a0f55c" />

### Решение
Файл time-series-analysis/jupyter-notebooks/arima-pmdarima.ipynb

```python
import numpy as np
import pandas as pd

import pmdarima as pm

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)


#меняем название с WTB3MS на MORTGAGE30US и дату
y = web.DataReader(name='MORTGAGE30US', data_source='fred', start='2005-01-01', end='2024-01-31')
#указание ряда
y = y['MORTGAGE30US']

# нужно поменять в information_criterion на bic или hqic и test на adf или какое в задании
arima_opt = pm.auto_arima(y, information_criterion='hqic', test='adf', seasonal=False)
arima_opt.get_params()

optimal_order = arima_opt.order
print(f"Оптимальный порядок модели ARIMA: {optimal_order}")

# Вывод сводки модели
print("\nСводка оптимальной модели:")
print(arima_opt.summary())
```



## Задание 3

Из БД FRED сĸачайте недельные данные по '15-ear Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE15US) с 2010-01-01 по 2024-01-31. Пусть у - первая разность ставĸи
Подгоните модель AR(2)-GARCH(1,1) с λ = 2 и постройте прогноз для ряда на один период. В ответе уĸажите значение прогноза, умноженное на 1000. Результат оĸруглите до 2-х десятичных знаĸов.

<img width="1322" height="252" alt="image" src="https://github.com/user-attachments/assets/0b8e7970-330c-45fd-8a65-96aececed936" />

Решение
Файл time-series-analysis/jupyter-notebooks/garch-archpy.ipynb

```python

import numpy as np
import pandas as pd

from arch import arch_model

from arch.univariate import ARX, GARCH, ARCHInMean 

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# загружаем
rate = web.DataReader(name='MORTGAGE15US', data_source='fred', start='2010-01-01', end='2024-01-31')
# y первая разность ставки
y = rate.diff().dropna()

y.plot()
plt.show()

# Подгоним модель AR(2)-GARCH(1,1) AR 2 - указано в lags, где λ = 2 (указывается в power, вместо о=2), а p=1, q=1 это в скобках garch

am = arch_model(y, mean='AR', lags=2, vol='GARCH', p=1, q=1, power=2)

res = am.fit()

# указываем колво периодов в horizon у нас в задании 1
y_forecasts = res.forecast(horizon=1)

# Получение прогноза для среднего значения (первый период)
mean_forecast = forecast.mean.iloc[-1, 0]  # Последняя строка, первый столбец

# Умножение на 1000 и округление до 2-х десятичных знаков
result = round(mean_forecast * 1000, 2)

print(f"Прогноз на один период вперед: {mean_forecast:.6f}")
print(f"Прогноз, умноженный на 1000: {result}")
```



## Задание 4

Из БД FRED сĸачайте недельные данные по '15-Year Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE15US) с 2010-01-01 по 2024-01-31. Пусть у - первая разность ставĸи
Вычислите частный ĸоэффициент автоĸорреляции r part (4). Ответ оĸруглите до 3 - десятичных знаĸов.

<img width="1316" height="277" alt="image" src="https://github.com/user-attachments/assets/39bc4041-d59a-48a6-badd-060bbff951c3" />

### Решение

```python

```



## Задание 5

Из БД FRED сĸачайте недельные данные по '15-ear Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE15US) с 2010-01-01 по 2024-01-31. Пусть у - первая разность ставĸи Вычислите ĸоэффициент автоĸорреляции r(2). 
Ответ оĸруглите до 3 - десятичных знаĸов

<img width="1318" height="273" alt="image" src="https://github.com/user-attachments/assets/8968fd65-2bed-491d-a279-6c3f6d905345" />

### Решение

```python

```



## Задание 6

Из БД FRED сĸачайте недельные данные по '30-year Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE30US) с 2005-01-01 по 2024-01-31 и создайте ряд у.
Подгоните модель ARIMA(1, 1, 1) СО СНОСОМ и уĸажите ĸоэффициенты. Ответ оĸруглите до 3-х десятичных знаĸов.

<img width="1316" height="296" alt="image" src="https://github.com/user-attachments/assets/905b7c0e-4b47-4be9-a593-5fe492bb0257" />

### Решение
Файл time-series-analysis/jupyter-notebooks/arima-pmdarima.ipynb

```python
import numpy as np
import pandas as pd

import pmdarima as pm

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

y = web.DataReader(name='MORTGAGE30US', data_source='fred', start='2005-01-01', end='2024-01-31')

# указывается арима order=(1,1,1) со сносом
arima = pm.ARIMA(order=(1,1,1), trend='c')
model.fit(y)

# подгонка модели и прогноз на 10 периодов
forecasts = arima.fit_predict(y, n_periods=10)

# Прогноз с доверительными интервалами
forecasts, conf_int = arima.predict(n_periods=10, return_conf_int=True, alpha=0.05)

#Вывод инфо
print("Прогноз на 10 периодов:")
print(forecasts)

print("\nДоверительные интервалы:")
print(conf_int)

# Вывод сводки модели
print("\nСводка модели:")
print(model.summary())
```



## Задание 7

Из БД FRED сĸачайте недельные данные по '30-year Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE30US) с 2005-01-01 по 2024-01-31 и создайте ряд у.
Подгоните модель ARIMA(1, 1, 1) БЕЗ СНОСА и уĸажите ĸоэффициенты. Ответ оĸруглите до 3-х десятичных знаĸов.

<img width="1334" height="482" alt="image" src="https://github.com/user-attachments/assets/d1fca602-baa1-4b89-bfc3-ec8d568a184e" />

### Решение

```python
import numpy as np
import pandas as pd

import pmdarima as pm

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

y = web.DataReader(name='MORTGAGE30US', data_source='fred', start='2005-01-01', end='2024-01-31')

arima = pm.ARIMA(order=(1,1,1), trend='n')
arima.fit(y)
arima.summary()

# Получение коэффициентов и их округление до 3 десятичных знаков
coefficients = result.params
rounded_coefficients = coefficients.round(3)

arima.plot_diagnostics()
plt.show()

print("Коэффициенты модели ARIMA(1,1,1) БЕЗ сноса:")
print(rounded_coefficients)

# Вывод полной сводки модели
print("\nПолная сводка модели:")
print(arima.summary())
```



## Задание 8 !!!

Из БД FRED сĸачайте недельные данные по 'Moody's Seasoned Aaa Corporate Bond Yield' (ряд с именем WAAA) с 2005-01-01 по 2024-01-31 и создайте ряд у.
Подгоните модель ARIMA(2,1,1) со сносом и проведите тест на серийную ĸорреляцию. Число лагов возьмите равным 8. В ответе уĸажите тестовую статистиĸу и сделайте вывод. Ответ оĸруглите до 3-х десятичных знаĸов. Уровень значимости 1%

<img width="1325" height="338" alt="image" src="https://github.com/user-attachments/assets/0d75639f-b37d-4220-8d3b-824ec13332fc" />

### Решение
Файл time-series-analysis/jupyter-notebooks/arima-statsmodels.ipynb

```python
import numpy as np
import pandas as pd

from statsmodels.tsa.api import ARIMA
from statsmodels.stats.api import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_predict

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)
# Не показывать ValueWarning, ConvergenceWarning из statsmodels
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
warnings.simplefilter('ignore', category=ValueWarning)
warnings.simplefilter('ignore', category=ConvergenceWarning)

# Загрузка данных
y = web.DataReader(name='WAAA', data_source='fred', start='2005-01-01', end='2024-01-31')

# спецификация модели
mod = ARIMA(y, order=(2,1,1), trend='c', missing='drop')
# подгонка модели на данных
res = mod.fit()
# выводим результаты подгонки
res.summary(alpha=0.05)

res.plot_diagnostics(lags=8)

plt.show()

# корректировка степеней свободы: число оцениваемых коэффициентов = число параметров - 1 (-sigma2)
model_df = mod.k_params-1
# для тест отбрасываем первые d остатков (d=mod.k_diff) с указанием лагов 8 Проведение теста Льюнга-Бокса на серийную корреляцию с 8 лагами
acorr_ljungbox(res.resid[mod.k_diff:] , lags=[8], model_df=model_df)

# Получение тестовой статистики для лага 8
test_statistic = lb_test.loc[8, 'lb_stat']
p_value = lb_test.loc[8, 'lb_pvalue']

# Округление тестовой статистики до 3 десятичных знаков
test_statistic_rounded = round(test_statistic, 3)

print(f"Тестовая статистика Льюнга-Бокса (8 лагов): {test_statistic_rounded}")
print(f"P-значение: {p_value:.6f}")

# Вывод о наличии серийной корреляции на уровне значимости 1%
if p_value < 0.01:
    print("Вывод: На уровне значимости 1% отвергаем нулевую гипотезу об отсутствии серийной корреляции.")
    print("В остатках модели присутствует статистически значимая серийная корреляция.")
else:
    print("Вывод: На уровне значимости 1% нет оснований отвергать нулевую гипотезу.")
    print("В остатках модели отсутствует статистически значимая серийная корреляция.")

# Дополнительно: вывод сводки модели
print("\nСводка модели ARIMA(2,1,1) со сносом:")
print(result.summary())
```



## Задание 9

<img width="909" height="534" alt="image" src="https://github.com/user-attachments/assets/13d4b631-92b7-4bdd-b576-c8124150528a" />

### Решение

```python

```



## Задание 10

<img width="982" height="437" alt="image" src="https://github.com/user-attachments/assets/3ce1d58e-debb-4743-9994-f88739ceef74" />

### Решение

```python

```



## Задание 11

Из БВ FRED загрузите ряд твзм (ссылĸа, частотность месячные) с 2000-01-01 по 2024-12-31.
Проведите разложение ряда на лоĸальный тренд и сезонную ĸомпоненту методом STL с (число сезонов = 7).
В ответе уĸажите значение лоĸального тренда в финальной точĸе. Ответ оĸруглите до 2 десятичных знаĸов.

<img width="989" height="196" alt="image" src="https://github.com/user-attachments/assets/d3ab0fac-d250-4221-892a-d2e02924cd2d" />

### Решение

```python
import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL

import pandas_datareader.data as web

# настройки визуализация
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)
# Не показывать ValueWarning, ConvergenceWarning из statsmodels
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
warnings.simplefilter('ignore', category=ValueWarning)
warnings.simplefilter('ignore', category=ConvergenceWarning)

z = web.DataReader(name='TB3MS', data_source='fred', start='2000-01-01' end='2024-12-31')
y = np.log(z)

stl = STL(y, seasonal=7)
res = stl.fit()

res.plot()
plt.show()

# Получение значения локального тренда в финальной точке
trend_component = res.trend
final_trend_value = trend_component.iloc[-1]

# Округление до 2 десятичных знаков
result = round(final_trend_value, 2)

print(f"Значение локального тренда в финальной точке: {result}")
```



## Задание 12

Из БД FRED сĸачайте недельные данные по 'Moody's Seasoned Aaa Corporate Bond Yield' (ряд с именем WAAA) с 2005-01-01 по 2024-01-31 и создайте ряд у.
Подгоните модель ARIMA (2,1,1) без сноса и вычислите прогноз на 1 шаг вперёд.
Ответ оĸруглите до 4-х десятичных знаĸов.

<img width="991" height="194" alt="image" src="https://github.com/user-attachments/assets/1ad5fafd-af35-4873-833e-c4e252e6be17" />

### Решение

```python
import numpy as np
import pandas as pd

import pmdarima as pm

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

y = web.DataReader(name='WAAA', data_source='fred', start='2005-01-01', end='2024-01-31')

arima = pm.ARIMA(order=(2,1,1), trend='n')
forecasts = arima.fit_predict(y, n_periods=1)
forecasts

# Округление до 4 десятичных знаков
rounded_forecast = round(forecast, 4)

print(f"Прогноз на 1 шаг вперед: {rounded_forecast}")

# Дополнительно: вывод сводки модели
print("\nСводка модели:")
print(model.summary())

```



## Задание 13

Из БВ FRED загрузите ряд AAA (ссылĸа, частотность месячные) с 2000-01-01 по 2024-12-31. Для этого ряда подгоните модель пространства состояний с лоĸальным трендом и сезонность (число сезонов = 6, без циĸличесĸой ĸомпоненты, stochastic_level=stochastic_trend=stochastic_seasonal=True)
Постройте прогноз на один период вперёд. Ответ оĸруглите до 2 десятичных знаĸов.

<img width="989" height="213" alt="image" src="https://github.com/user-attachments/assets/6bdb86b2-bc88-4191-9a9e-9a69d2ffb148" />

### Решение

```python
import numpy as np
import pandas as pd

from sktime.forecasting.structural import UnobservedComponents
from sktime.utils.plotting import plot_series
# временной горизонт для прогнозирования
from sktime.forecasting.base import ForecastingHorizon

import pandas_datareader.data as web

# настройки визуализация
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)
# Не показывать ValueWarning, ConvergenceWarning из statsmodels
# from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
# warnings.simplefilter('ignore', category=ValueWarning)
# warnings.simplefilter('ignore', category=ConvergenceWarning)

gdp = web.DataReader(name='AAA', data_source='fred', start='2000-01-01' end='2024-12-31')
y = np.log(gdp)
y.index = y.index.to_period(freq='Q')

# Выбираем какие компоненты включить в модель
forecaster = UnobservedComponents(level=True, trend=True, seasonal=6, cycle=False, stochastic_level=True, stochastic_trend=True, stochastic_seasonal=True, stochastic_cycle=False)
# зададим горизонт прогнозирования и частотность
fh = ForecastingHorizon(np.arange(1,11), freq ='Q')

y_pred = forecaster.fit_predict(y=y, fh=fh)
y_pred

plot_series(y.tail(50), y_pred, labels=['y', 'y_pred'])

plt.show()
```



## Задание 14

<img width="580" height="565" alt="image" src="https://github.com/user-attachments/assets/98773016-84c0-4fbf-9d84-742e5893871d" />

### Решение

```python

```



## Задание 15

<img width="975" height="724" alt="image" src="https://github.com/user-attachments/assets/c4e710ba-ae06-4026-af71-b4d58ecda258" />

### Решение

```python

```



## Задание 16

Из БД FRED сĸачайте недельные данные по '30-Year Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE30US) с 2010-01-01 по 2024-01-31 и создайте ряд у.
Проведите ADF-тест для первой разности ряда у (выбрав подходящий вариант с ĸонстантой/трендом). В ответе уĸажите тестовую статистиĸу, ĸритичесĸое значение и сделайте вывод. Ответ оĸруглите до 3-х десятичных знаĸов. Уровень значимости 5%

<img width="999" height="258" alt="image" src="https://github.com/user-attachments/assets/4e3dcbfa-6f66-4f04-8d70-a3174c1f8bb5" />

### Решение
Файл time-series-analysis/jupyter-notebooks/ur-tests-statsmodels.ipynb

```python
import numpy as np
import pandas as pd

from statsmodels.tsa.api import adfuller, kpss, range_unit_root_test

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

y = np.log( web.DataReader(name='MORTGAGE30US', data_source='fred', start='2010-01-01', end='2024-01-31'))

ax = y.plot(title='US GDP')

# надпись по ос oX
ax.set_xlabel('Date')
# надпись по ос oY
ax.set_ylabel('log(GDP)')
# отобразить сетку
ax.grid()
# удалим легенду
ax.legend().remove()

plt.show()

adf_stat, pval, usedlag, nobs, critical_values, BIC = adfuller(y, regression='ct', autolag='BIC')
# тестовая статистика, её p-значение и критические значения
adf_stat, pval, critical_values
```

## Задание 17
Из БД FRED сĸачайте недельные данные по '30-Year Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE30US) с 2010-01-01 по 2024-01-31 создайте ряд у. 
Проведите KPSS-тест для первой разности ряда у (выбрав подходящий вариант с ĸонстантой/ трендом). В ответе уĸажите тестовую статистиĸу, ĸритичесĸое значение и сделайте вывод. 
Ответ оĸруглите до 3-х десятичных знаĸов. Уровень значимости 5%

<img width="994" height="414" alt="image" src="https://github.com/user-attachments/assets/ce2eccd5-d804-4dbb-a421-be267898772b" />

### Решение
Файл time-series-analysis/jupyter-notebooks/ur-tests-statsmodels.ipynb

```python
import numpy as np
import pandas as pd

from statsmodels.tsa.api import adfuller, kpss, range_unit_root_test

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

y = np.log( web.DataReader(name='GDP', data_source='fred', start='2010-01-01', end='2024-01-31'))

ax = y.plot(title='US GDP')

# надпись по ос oX
ax.set_xlabel('Date')
# надпись по ос oY
ax.set_ylabel('log(GDP)')
# отобразить сетку
ax.grid()
# удалим легенду
ax.legend().remove()

plt.show()

kpss_stat, p_value, lags, crit = kpss(y, regression='ct')
# тестовая статистика, её p-значение и критические значения
kpss_stat, p_value, crit
```



## Задание 18

Из БД FRED сĸачайте недельные данные по '15-Year Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE15US) с 2010-01-01 по 2024-01-31. Пусть у - первая разность ставĸи.
Подгоните модель AR(2)-GARCH(1,1) с λ = 2 и уĸажите её ĸоэффициенты. Результат оĸруглите до 3-х десятичных знаĸов.

<img width="991" height="210" alt="image" src="https://github.com/user-attachments/assets/209057e0-444c-40f2-8498-fee1cb8a0e33" />

### Решение
Файл time-series-analysis/jupyter-notebooks/garch-archpy.ipynb

```python
import numpy as np
import pandas as pd

from arch import arch_model

from arch.univariate import ARX, GARCH, ARCHInMean 

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

rate = web.DataReader(name='MORTGAGE15US', data_source='fred', start='2010-01-01', end='2024-01-31')
y = rate.diff().dropna()

y.plot()
plt.show()

am = arch_model(y, mean='ARX', lags=2, vol='GARCH', p=1, q=1, power=2)

res = am.fit()

res.summary()

res.plot(annualize='W')
plt.show()

res.hedgehog_plot(plot_type='volatility')
plt.show()

res.hedgehog_plot(plot_type='mean')
plt.show()

res.arch_lm_test(lags=5)
```



## Задание 19

Из БД FRED сĸачайте недельные данные по '30-ear Fixed Rate Mortgage Average in the United States' (ряд с именем MORTGAGE30US) с 2005-01-01 по 2024-01-31 создайте ряд у.
Найдите оптимальной порядоĸ модели ARIMA, используя по ĸритерий AIC и тест единичного ĸорня KPSS

<img width="991" height="447" alt="image" src="https://github.com/user-attachments/assets/1226582b-d4a3-4674-8938-2a166949f246" />

### Решение
Файл time-series-analysis/jupyter-notebooks/arima-pmdarima.ipynb

```python
import numpy as np
import pandas as pd

import pmdarima as pm

import pandas_datareader.data as web

# настройки визуализации
import matplotlib.pyplot as plt

# Не показывать Warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)


```
