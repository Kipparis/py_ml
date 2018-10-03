#!/usr/bin/env python3 

import pandas as pd
# создаем простой набор данных с характеристиками пользователей
data = {'Name': ["John", "Anna", "Peter", "Linda"],
'Location' : ["New York", "Paris", "Berlin", "London"], 'Age' : [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)
# IPython.display позволяет "красиво напечатать" датафреймы 
# в Jupyter notebook
display(data_pandas)