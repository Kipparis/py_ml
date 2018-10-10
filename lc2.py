# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt # библиотека для отрисовки графиков
import numpy as np # импортируем numpy для создания своего датасета 
from sklearn import linear_model, model_selection # импортируем линейную модель для обучения и библиотеку для разделения нашей выборки
# from sklearn import cross_validation, datasets, metrics
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
# Импортируем библиотеки для валидации, создания датасетов, и метрик качества



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
regr = linear_model.LinearRegression(n_jobs=2)
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
print("\nСоздаём выборку ",X.shape, y.shape, coef)

# X - признаки, у - нахождение объекта которому принадлежат
# plt.scatter(X[:, 0], y, color='r')
# plt.scatter(X[:, 1], y, color='b')

# plt.show()

# Разделяем выборку для обучения и тестирования
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Создаём классификатор с линейной регрессией
linear_regressor = linear_model.LinearRegression(n_jobs=2)
# Тренируем её
linear_regressor.fit(X_train, y_train)
# Делаем предсказания
y_pred = linear_regressor.predict(X_test)
# Выводим ошибки
print(metrics.mean_absolute_error(y_test, y_pred))

print("right \t\t", coef[0], coef[1])
print("incorrect\t", linear_regressor.coef_[1], linear_regressor.intercept_)

linear_scoring = model_selection.cross_val_score(linear_regressor, X, y, cv=10)
print ('Средняя ошибка: {}, Отклонение: {}'.format(linear_scoring.mean(), linear_scoring.std()))

# Создаем свое тестирование на основе абсолютной средней ошибкой
scorer = metrics.make_scorer(metrics.mean_absolute_error)

linear_scoring = model_selection.cross_val_score(linear_regressor, X, y, scoring=scorer, cv=10)
print ('Средняя ошибка: {}, Отклонение: {}'.format(linear_scoring.mean(), linear_scoring.std()))

print ("y = {:.2f}*x1 + {:.2f}*x2".format(coef[0], coef[1]))

print ("y = {:.2f}*x1 + {:.2f}*x2 + {:.2f}".format(linear_regressor.coef_[0], 
                                                  linear_regressor.coef_[1], 
                                                  linear_regressor.intercept_))

# Посмотрим качество обучения
print('Оценка дисперсии:: %.2f' % linear_regressor.score(X_test, y_test))

print("right \t\t", coef[0], coef[1])
print("incorrect\t", linear_regressor.coef_[0], linear_regressor.intercept_)
