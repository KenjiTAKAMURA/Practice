#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


x_train = np.arange(20).reshape(10,2)
t_train = np.arange(10).reshape(10,1)
batch_size = 3


# In[9]:


# ランダムにミニバッチへ分割するために、インデックスをランダムに並び替える

import math
perm = np.random.permutation(len(x_train)) ###### 問3.1 ######

# batch_size ごとにデータを読み込んで学習させる
#12個をランダムにシャッフルし、最初の5個、次の5個、残り2個の3つのバッチで学習させる。
for idx in np.split(perm, [i*batch_size for i in range(1, math.ceil(len(perm)/batch_size))]): ###### 問3.2 ######
    print(x_train[idx]) ###### 問3.3 ######
    print(t_train[idx]) ###### 問3.4 ######


# In[24]:


# ランダムにミニバッチへ分割するために、インデックスをランダムに並び替える

import math
perm = np.random.permutation(len(x_train)) ###### 問3.1 ######

# batch_size ごとにデータを読み込んで学習させる
#12個をランダムにシャッフルし、最初の5個、次の5個、残り2個の3つのバッチで学習させる。
for idx in np.split(perm, [i*batch_size for i in range(1, math.ceil(len(perm)/batch_size))]): ###### 問3.2 ######

    if x_train[idx].shape[0] < batch_size:
        break
    print(x_train[idx]) ###### 問3.3 ######
    print(t_train[idx]) ###### 問3.4 ######


# In[11]:


[i*batch_size for i in range(1, math.ceil(len(perm)/batch_size))]


# In[21]:


a = np.array([[1,2],[3,4]])
a.shape[0]


# In[ ]:




