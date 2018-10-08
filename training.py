from sklearn.datasets   import make_circles
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import mglearn
import matplotlib.pyplot as plt

import numpy as np

X, y = make_circles()
print("Circles samples")

plt.scatter(X[:, 0], X[:, 1], marker="H", c=y)
# plt.show()

# Определяем кол-во графиков
graphics = 9
rows = 3
columns = graphics // rows

# Создаём три графика с тремя осями
fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 3*rows))

# Разделяем набор данных на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)

neighbours_arr = [1+3*(x-1) for x in range(1,graphics+1)]
print(neighbours_arr)
for n_neighbours, ax in zip(neighbours_arr, axes.reshape(-1)):
    # Первый арг в цикле - количество соседей в классификаторе
    # Второй - оси в сабплоте
    clf = KNeighborsClassifier(n_neighbors=n_neighbours).fit(X_train,y_train)
    # С помощью mglearn делаем график отдельно для каждого классификатора
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_pred, ax=ax) 
    ax.set_title("{} сосед(ей)".format(n_neighbours))
    ax.set_xlabel("признак 0")
    ax.set_ylabel("признак 1")
    ax.legend(['{:.2f}'.format(accuracy)])


fig.subplots_adjust(wspace=0.5, hspace = 0.5)
axes.reshape(rows, columns)
plt.show()



# # Разделяем данные на обучающие и тестовые
# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=2)

# training_accuracy   = [] # Точность обучения
# test_accuracy       = [] # Точность теста

# # Устанавливаем количество соседей в [0, 20]
# neighbours_settings = range(1, 11)
# # Для каждого кол-ва соседей создаём свою классификатор и выясняем точность 
# for n_neighbours in neighbours_settings:
#     # Строим модель
#     reg = KNeighborsClassifier(n_neighbors=n_neighbours)
#     reg.fit(X_train, y_train)
#     # Запишем обучающую точность
#     training_accuracy.append(reg.score(X_train, y_train))
#     # Запишем тестовую точность
#     test_accuracy.append(reg.score(X_test, y_test))

# plt.plot(neighbours_settings, training_accuracy, label="Обучающая точность")
# plt.plot(neighbours_settings, test_accuracy, label="Тестовая точность")
# plt.xlabel("n_соседей")
# plt.ylabel("Точность")
# plt.legend()
# plt.show()
