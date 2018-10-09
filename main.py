# -*- coding: utf-8 -*-

from random import randint

from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split

import mglearn

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, model_selection

import numpy as np

X, y = make_blobs(random_state=1, n_samples=100, n_features=2, centers=4)

print("\nX[:, 0] -->\n{}".format(X[:, 0]))
print("\nX[:, 1] -->\n{}".format(X[:, 1]))
print(X)

print("X type: ", type(X))
print("y type: ", type(y))

newX = np.ndarray(shape=(2,2), dtype=float, order='F')

# Создаём массив ndarray
rows = 10
columns = 2
newX = np.array([
    [x+columns*y for x in range(columns)] for y in range(rows)
])
print(newX)

print("\nnewX type: ", type(newX))

newY = [randint(0, 2) for y in range(rows * columns)]
print("newY is {}".format(newY))

# plt.scatter(newX[:, 0], newX[:, 1], marker="H", c=newY)
# plt.show()

print("\nnewX[:, 0] -->\n{}".format(newX[:, 0]))
print("\nnewX[:, 1] -->\n{}".format(newX[:, 1]))


# print(y)
# plt.scatter(X[:, 0], X[:, 1], marker="H", c=y)
# plt.show()

# Проверяем как plt будет отображать кастомные массивы

# knn = KNeighborsClassifier(n_neighbors=1)

# iris_dataset = load_iris()

# print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))

# print(iris_dataset['DESCR'][:193] + "\n...")

# print("\nНазвания ответов: {}".format(iris_dataset['target_names']))

# print("\nНазвание празинаков:\n{}".format(iris_dataset['feature_names']))

# print("\nТип массива data: {}".format(type(iris_dataset['data'])))

# print("\nФорма массива data: {}".format(iris_dataset['data'].shape))

# print("\nПервые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))

# print("\nФорма массива target: {}".format(iris_dataset['target'].shape))

# print("Ответы :\n{}".format(iris_dataset['target']))

# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# print("\nФорма массива X_train: {}".format(X_train.shape))
# print("Форма массива y_train: {}".format(y_train.shape))

# print("\nФорма массива  X_test: {}".format(X_test.shape))
# print("Форма массива  y_test: {}".format(y_test.shape))

# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])
# # создаем матриуц рессеяния из dataframe, цвет точек задается с помощью y_train
# grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# # plt.title('scatter-matrix')
# plt.show()

# knn.fit(X_train, y_train)

# print(knn)

# X_new = np.array([[5, 2.9, 1, 0.2]])
# print("Форма массива X_new:\n{}".format(X_new.shape))

# prediction = knn.predict(X_new)
# print("\nПрогноз:\n{}".format(prediction))
# print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction]))

# y_pred = knn.predict(X_test)
# print("Прогнозы для тестового набора:\n{}".format(y_pred))

# print("Точность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))

