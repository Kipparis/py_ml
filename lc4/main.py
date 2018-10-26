# -*- coding: utf-8 -*-

from __future__ import division, print_function


import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
import graphviz 

from matplotlib import pyplot as plt

# Первый класс
np.random.seed(7)
train_data = np.random.normal(size=(100, 2))
train_labels = np.zeros(100)

# Второй класс
train_data = np.r_[train_data, np.random.normal(size=(100, 2), loc=2)]
train_labels = np.r_[train_labels, np.ones(100)]

# Функция для визуализации

def get_grid(data, eps=0.01):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, eps), np.arange(y_min, y_max, eps))

def show_tree(tree, features=1):
    export_graphviz(tree, feature_names=['f'+str(i) for i in range(1,1+features)], 
                    out_file='small_tree.dot', filled=True)
    print("feature_names is ", ['f'+str(i) for i in range(1,1+features)])
    s = graphviz.Source.from_file('small_tree.dot')
    s.view()

    # export_graphviz(tree, feature_names=['Возраст'], 
    #             out_file='age_tree.dot', filled=True)
    # with open("age_tree.dot") as f:
    #     dot_graph = f.read() 
    # graphviz.Source(dot_graph)


clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)

clf_tree.fit(train_data, train_labels)

# # Немного кода для отображения разделяющей поверхности
# xx, yy = get_grid(train_data)
# predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
# plt.pcolormesh(xx, yy, predicted, cmap='cool')
# print(xx.shape, yy.shape)
# print(xx.ravel(), xx.ravel().shape, "\n\n", yy.ravel(), yy.ravel().shape)
# plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, s=100,
#             cmap='cool', edgecolors='black', linewidths=1.5)


# with open("small_tree.dot") as f:
#     dot_graph = f.read()
# graphviz.Source(dot_graph)


# plt.show()
data2 = pd.DataFrame({'Возраст':  [17,64,18,20,38,49,55,25,29,31,33], 
                      'Зарплата': [25,80,22,36,37,59,74,70,33,102,88], 
             'Невозврат кредита': [1,0,1,0,1,0,0,1,1,0,1]})


data_sorted = data2.sort_values('Возраст')

print(data_sorted)

tree = DecisionTreeClassifier(random_state=17)
# tree.fit(data['Возраст'].values.reshape(-1, 1), data["Невозврат кредита"].values)
tree.fit(data2[['Возраст', 'Зарплата']].values, data2['Невозврат кредита'].values)

# show_tree(tree, features=2)

# Немного кода для отображения разделяющей поверхности

xx, yy = get_grid(data2[['Возраст', 'Зарплата']].values)

print(data2[['Возраст','Зарплата']].values[:, 0], 
        data2[['Возраст','Зарплата']].values[:, 1],)

predicted = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='cool')
plt.scatter(data2['Возраст'].values, data2['Зарплата'].values, 
            c=data2['Невозврат кредита'], cmap='cool', 
            edgecolors='black', linewidths=1.5)

plt.show()

# print()