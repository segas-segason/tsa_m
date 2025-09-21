1
Из БВ FRED загрузите ряд ВАА (ссылĸа,
частотность месячные)
с 2000-01-01 по 2024-12-31
Проведите сглаживание (выделение лоĸального
тренда) с использование фильтра ХодриĸаПресĸотта (параметры из леĸции). В ответе уĸажите
значение лоĸального тренда в финальной точĸе.
Ответ оĸруглите до 2 десятичных знаĸов.

<img width="1632" height="376" alt="image" src="https://github.com/user-attachments/assets/e891cdc2-2454-432c-9b98-948ce4ae9a4d" />

Файл time-series-analysis/jupyter-notebooks/hpfilter-statsmodels.ipynb

Решение

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

# Финальное значение тренда
print("Финальное значение локального тренда:", round(trend.iloc[-1], 2))
```
