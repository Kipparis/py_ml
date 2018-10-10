# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt # библиотека для отрисовки графиков
import numpy as np # импортируем numpy для создания своего датасета 
from sklearn import linear_model, model_selection # импортируем линейную модель для обучения и библиотеку для разделения нашей выборки
from sklearn.model_selection import train_test_split
# Импортируем библиотеки для валидации, создания датасетов, и метрик качества
from sklearn import cross_validation, datasets, metrics



X = np.random.randint(100,size=(500, 1)) # создаем вектор признаков, вектора так как у нас один признак 
y = np.random.normal(np.random.randint(300,360,size=(500, 1))-X) # создаем вектор ответом  
print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))
# plt.scatter(X, y) # рисуем график точек
# plt.xlabel('Расстояние до кафе в метрах') # добавляем описание для оси x
# plt.ylabel('Количество заказов')# добавляем описание для оси y
# plt.show()

# Делим данные на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Создаём классификатор
regr = linear_model.LinearRegression()
# Обучаем модель
regr.fit(X_train, y_train)

# Коэффиценты которые установила модель
print('Коэфицент: \n', regr.coef_)
# Средний квадрат ошибки
print('Средний квадрат ошибки: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2))
# Оценка дисперсии: 1 - идеальное предсказание. Качество предсказания
print('Оценка дисперсии: %.2f' % regr.score(X_test, y_test))

# Посмотрим на получившуюся функцию
print("y = {:.2f}*x + {:.2f}".format(regr.coef_[0][0], regr.intercept_[0]))

# Посмотрим как наша модель предсказывает тестовые данные
# plt.scatter(X_test, y_test, color='black') # Рисуем график точек
# plt.plot(X_test, regr.predict(X_test), color='blue') # Рисуем график линейной регрессии
# plt.show()

# Создаём датасет с избыточной информацией
X, y, coef = datasets.make_regression(n_features=2, n_informative=1, n_targets=1,
                                        noise=5., coef=True, random_state=2)
# Поскольку у нас есть два признака, для отрисовки надо их разделить на две части
data_1, data_2 = [], []