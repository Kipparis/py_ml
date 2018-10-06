# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt # библиотека для отрисовки графиков
import numpy as np # импортируем numpy для создания своего датасета 
from sklearn import linear_model, model_selection # импортируем линейную модель для обучения и библиотеку для разделения нашей выборки



X = np.random.randint(100,size=(300, 1)) # создаем вектор признаков, вектора так как у нас один признак 
y = np.random.normal(np.random.randint(100,400,size=(300, 1))-X) # создаем вектор ответом  
plt.scatter(X, y) # рисуем график точек
plt.xlabel('Расстояние до кафе в метрах') # добавляем описание для оси x
plt.ylabel('Количество заказов')# добавляем описание для оси y
plt.show()