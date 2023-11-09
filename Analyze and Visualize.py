#!/usr/bin/env python
# coding: utf-8

# # I. Numpy

# ### Импортируйте NumPy

# In[2]:


import numpy as np


# ### Создайте одномерный массив размера 10, заполненный нулями и пятым элемент равным 1. Трансформируйте в двумерный массив.

# In[4]:


aray = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
aray.reshape((5,2))


# ### Создайте одномерный массив со значениями от 10 до 49 и разверните его (первый элемент становится последним). Найдите в нем все четные элементы.

# In[5]:


aray = np.arange(10, 50)
a = aray[::2]
b = a[::-1]
b


# ### Создайте двумерный массив 3x3 со значениями от 0 до 8

# In[6]:


aray = np.arange(0, 9)
aray.reshape(3, 3)


# ### Создайте массив 4x3x2 со случайными значениями. Найти его минимум и максимум.

# In[7]:


aray = np.random.randint(0, 1000, size = (4, 3, 2))
column_max = aray.max()
column_min = aray.min()
column_max
column_min


# ### Создайте два двумерных массива размерами 6x4 и 4x3 и произведите их матричное умножение. 

# In[3]:


aray_1 = ([1, 2, 4, 3, 6, 5], 
          [2, 1, 5, 3, 4, 6], 
          [3, 1, 4, 2, 5, 6], 
          [4, 2, 6, 1, 5, 3])
aray_2 = ([1, 2, 4, 3], 
          [2, 1, 5, 3], 
          [3, 1, 4, 2])
np.outer(aray_1, aray_2)


# ### Создайте случайный двумерный массив 7x7, найти у него среднее и стандартное оклонение. Нормализуйте этот массив.

# In[8]:


aray_1 = np.random.rand(7, 7)
print(np.average(aray_1))
print(np.std(aray_1))
print(np.linalg.norm(aray_1))


# # II. Pandas

# ### Импортируйте: pandas, matplotlib, seaborn

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Загрузите датасет Tips из набора датасетов seaborn

# In[4]:


tips_raw = pd.read_csv('tips.csv')


# ### Посмотрите на первые 5 строчек

# In[5]:


display(tips_raw.iloc[1:6])


# ### Узнайте сколько всего строчек и колонок в данных

# In[13]:


rows = len(tips_raw.axes[0]) 
cols = len(tips_raw.axes[1]) 
print('rows:', rows)
print('cols:', cols)


# ### Проверьте есть ли пропуски в данных

# In[7]:


tips_raw.isnull().sum().sum()


# ### Посмотрите на распределение числовых признаков

# In[ ]:





# ### Найдите максимальное значение 'total_bill'

# In[8]:


tips_raw.head(11)
max_bill = tips_raw['total_bill'].max()
max_bill


# ### Найдите количество курящих людей

# In[9]:


smokers = tips_raw[['smoker']]
smokers['smoker'].hist()


# ### Узнайте какой средний 'total_bill' в зависимости от 'day'

# In[10]:


count=pd.crosstab(tips_raw.total_bill, tips_raw.day)
c1 = count.mean()
c1.T.plot(kind='bar')
plt.show()


# ### Отберите строчки с 'total_bill' больше медианы и узнайте какой средний 'tip' в зависимости от 'sex'

# In[11]:


res = tips_raw.loc[tips_raw["total_bill"] > tips_raw["total_bill"].median()]
display(res)
mean_f = tips_raw.loc[tips_raw['sex'] == 'Female', 'tip'].mean()
mean_m = tips_raw.loc[tips_raw['sex'] == 'Male', 'tip'].mean()
display('Средний tip Male', mean_f)
display('Средний tip Female', mean_m)


# ### Преобразуйте признак 'smoker' в бинарный (0-No, 1-Yes)

# In[28]:


s_bin = pd.get_dummies(tips_raw['smoker'])
df_two = pd.concat((s_bin, tips_raw), axis=1)
df_two = df_two.drop(["smoker"], axis=1)
df_two = df_two.drop(["No"], axis=1)
result = df_two.rename(columns={"Yes": "smoker"})
print(result)


# # III. Visualization

# ### Постройте гистограмму распределения признака 'total_bill'

# In[29]:


t_b = tips_raw[['total_bill']]
t_b.hist()


# ### Постройте scatterplot, представляющий взаимосвязь между признаками 'total_bill' и 'tip'

# In[32]:


sns.regplot(x='total_bill', 
            y='tip', 
            data=tips_raw)
plt.xlabel('total_bill')
plt.ylabel('tip')


# ### Постройте pairplot

# In[35]:


sns.pairplot(tips_raw)
plt.show()


# ### Постройте график взаимосвязи между признаками 'total_bill' и 'day'

# In[20]:


sns.set(style ="darkgrid")
sns.stripplot(x ="total_bill", y ="day", data = tips_raw); 


# ### Постройте две гистограммы распределения признака 'tip' в зависимости от категорий 'time'

# In[15]:


sns.barplot(x='tip', y='time', data=tips_raw)


# # Постройте два графика scatterplot, представляющих взаимосвязь между признаками 'total_bill' и 'tip' один для Male, другой для Female и раскрасьте точки в зависимоти от признака 'smoker'

# In[25]:


sns.catplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'sex', data = tips_raw)


# ## Сделайте выводы по анализу датасета и построенным графикам. По желанию можете продолжить анализ данных и также отразить это в выводах.

# In[ ]:


# Чем больше счёт, тем больше чаевые.
# В выходные больше заказов, чем в будни
# В среднем чаевые больше в вечернее время работы
# Мужчины оставляют больше чаевых и курящие люди оставляют чаевых больше, чем не курящие

