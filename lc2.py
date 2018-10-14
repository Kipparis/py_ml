# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt # библиотека для отрисовки графиков
import matplotlib.cm as cm

import numpy as np # импортируем numpy для создания своего датасета 
from sklearn import linear_model, model_selection # импортируем линейную модель для обучения и библиотеку для разделения нашей выборки
# from sklearn import cross_validation, datasets, metrics
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split

import mglearn
# Импортируем библиотеки для валидации, создания датасетов, и метрик качества



# X = np.random.randint(100,size=(500, 1)) # создаем вектор признаков, вектора так как у нас один признак 
# y = np.random.normal(np.random.randint(300,360,size=(500, 1))-X) # создаем вектор ответом  
# print("Shape of X: {}".format(X.shape))
# print("Shape of y: {}".format(y.shape))
# # plt.scatter(X, y) # рисуем график точек
# # plt.xlabel('Расстояние до кафе в метрах') # добавляем описание для оси x
# # plt.ylabel('Количество заказов')# добавляем описание для оси y
# # plt.show()

# # Делим данные на обучающую и тестовую
# X_train, X_test, y_train, y_test = train_test_split(X,y)

# # Создаём классификатор
# regr = linear_model.LinearRegression(n_jobs=2)
# # Обучаем модель
# regr.fit(X_train, y_train)

# # Коэффиценты которые установила модель
# print('Коэфицент: \n', regr.coef_)
# # Средний квадрат ошибки
# print('Средний квадрат ошибки: %.2f' % np.mean((regr.predict(X_test) - y_test) ** 2))
# # Оценка дисперсии: 1 - идеальное предсказание. Качество предсказания
# print('Оценка дисперсии: %.2f' % regr.score(X_test, y_test))

# # Посмотрим на получившуюся функцию
# print("y = {:.2f}*x + {:.2f}".format(regr.coef_[0][0], regr.intercept_[0]))

# Посмотрим как наша модель предсказывает тестовые данные
# plt.scatter(X_test, y_test, color='black') # Рисуем график точек
# plt.plot(X_test, regr.predict(X_test), color='blue') # Рисуем график линейной регрессии
# plt.show()



# # Посмотрим на данные
# plt.scatter(X[:, 0], y, color='r')
# plt.scatter(X[:, 1], y, color='g')
# plt.scatter(X[:, 2], y, color='b')

# plt.show()



# СОпределяем кол-во графиков
graphics = 9
rows = 3
columns = graphics // rows

# создаём графики
fig, axes = plt.subplots(rows, columns, figsize=(6*columns, 6*rows))

features_count = [i+2 for i in range(0, graphics)]
for feature_count, ax in zip(features_count, axes.reshape(-1)):
    # Создаём данные
    X, y, coef = datasets.make_regression(n_features=feature_count, n_informative=1, n_targets=1,
                                        noise=5., coef=True, random_state=2)
    # Добавляем данные на график
    # Создаём палитру цветов для графиков
    colors = cm.rainbow(np.linspace(0, 1, feature_count))
    for j in range(0, feature_count):
        ax.scatter(X[:, j], y, color=colors[j])
        # print(j)
    
    ax.set_title("{} ненужных признаков".format(feature_count-1))

    # Делаем сплит  данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    
    # Создаём тренировочную модель
    lg = linear_model.LinearRegression(n_jobs=2)
    # Тренируем её
    lg.fit(X_train, y_train)
    # Создаём предсказание
    y_pred = lg.predict(X_test)
    # print(y_pred)

    legend=[]

    # Добавляем ошибки
    scorer = metrics.make_scorer(metrics.mean_absolute_error)
    linear_scoring = model_selection.cross_val_score(lg, X, y, scoring=scorer, cv=10)
    legend.append('Средняя ошибка: {}'.format(linear_scoring.mean()))
    legend.append('Отклонение: {}'.format(linear_scoring.std()))

    # разность коэффицентов
    # Выводим коэффиценты модели
    coef_dif_sum = 0.
    for coeff, lgcoef in zip(coef, lg.coef_):
        coef_dif_sum += abs(coeff - lgcoef)

    print("Maximum : {} \t Minimum : {}".format(np.amax(X), np.amin(X)))
    print("Coef from model with {} features".format(feature_count))
    print("Coefs {} \t Coefs from classifier {} \n".format(coef,lg.coef_))

    legend.append("Среднее отклонение коэф. {}".format(coef_dif_sum / len(coef)))

    # Рисуем линию для определения работы алгоритма

    ax.legend(legend, loc=3)

plt.show()



# X - признаки, у - нахождение объекта которому принадлежат
# plt.scatter(X[:, 0], y, color='r')
# plt.scatter(X[:, 1], y, color='b')

# plt.show()

# # Разделяем выборку для обучения и тестирования
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # Создаём классификатор с линейной регрессией
# linear_regressor = linear_model.LinearRegression(n_jobs=2)
# # Тренируем её
# linear_regressor.fit(X_train, y_train)
# # Делаем предсказания
# y_pred = linear_regressor.predict(X_test)
# # Выводим ошибки
# print(metrics.mean_absolute_error(y_test, y_pred))

# print("\nОшибки используя скросс валидацию:")
# linear_scoring = model_selection.cross_val_score(linear_regressor, X, y, cv=10)
# print ('Средняя ошибка: {}, Отклонение: {}'.format(linear_scoring.mean(), linear_scoring.std()))

# print("\nСоздаем свое тестирование на основе абсолютной средней ошибкой")
# # Создаем свое тестирование на основе абсолютной средней ошибкой
# scorer = metrics.make_scorer(metrics.mean_absolute_error)

# print("Ошибки используя кросс валидацию со своим правилом подсчёта")
# linear_scoring = model_selection.cross_val_score(linear_regressor, X, y, scoring=scorer, cv=10)
# print ('Средняя ошибка: {}, Отклонение: {}'.format(linear_scoring.mean(), linear_scoring.std()))

# print ("y = {:.2f}*x1 + {:.2f}*x2".format(coef[0], coef[1]))

# print ("y = {:.2f}*x1 + {:.2f}*x2 + {:.2f}".format(linear_regressor.coef_[0], 
#                                                   linear_regressor.coef_[1], 
#                                                   linear_regressor.intercept_))

# # Посмотрим качество обучения
# print('Оценка дисперсии:: %.2f' % linear_regressor.score(X_test, y_test))

# print("right \t\t", coef[0], coef[1])
# print("incorrect\t", linear_regressor.coef_[0], linear_regressor.intercept_)
