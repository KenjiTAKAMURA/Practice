#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[39]:


#畳み込み演算
#ストライド１
#パディングなし
def convolve1(array, kernel):
    a_h, a_w = array.shape
    k_h, k_w = kernel.shape
    row, column = a_h - (k_h - 1), a_w - (k_w - 1)
    output = np.zeros((row, column))
    
    for i in range(row):
        for j in range(column):
            output[i, j] = np.sum(array[i:i+k_h,j:j+k_w]*kernel)
    
    return output


# In[40]:


array = np.array([[i for i in range(1, 5)],
                  [i for i in range(6, 10)],
                  [i for i in range(11, 15)],
                  [i for i in range(16, 20)]])
kernel = np.arange(1, 5).reshape(2,2)
print(array)
print(kernel)
convolve1(array, kernel)


# In[78]:


#畳み込み演算
#ストライド，パディング指定可能
def convolve2(array, kernel, stride, padding):
    a_h, a_w = array.shape
    k_h, k_w = kernel.shape
    row = int(((a_h +2*padding) - k_h)/stride)+1
    column = int(((a_w +2*padding) - k_w)/stride)+1
    output = np.zeros((row, column))
    
    #arrayにパディングを付ける
    array_padding = np.zeros((a_h+2*padding, a_w+2*padding))
    array_padding[padding:padding+a_h,padding:padding+a_w]=array
    print(array_padding)
    #畳み込みの計算
    for i in range(row):
        _i = i*stride
        for j in range(column):
            _j = j*stride
            output[i, j] = np.sum(array_padding[_i:_i+k_h,_j:_j+k_w]*kernel)
    
    return output


# In[83]:


array = np.array([[i for i in range(1, 5)],
                  [i for i in range(6, 10)],
                  [i for i in range(11, 15)],
                  [i for i in range(16, 20)]])
kernel = np.arange(1, 5).reshape(2,2)
print(array)
print(kernel)

convolve2(array, kernel, 1, 0)


# In[96]:


#マックスプーリング
#ストライド，パディング指定可能

def max_pooling(array, kernel_shape, stride, padding):
    a_h, a_w = array.shape
    k_h, k_w = kernel_shape
    row = int(((a_h +2*padding) - k_h)/stride)+1
    column = int(((a_w +2*padding) - k_w)/stride)+1
    output = np.zeros((row, column))
    
    #arrayにパディングを付ける
    array_padding = np.zeros((a_h+2*padding, a_w+2*padding))
    array_padding[padding:padding+a_h,padding:padding+a_w]=array
    print(array_padding)
    #畳み込みの計算
    for i in range(row):
        _i = i*stride
        for j in range(column):
            _j = j*stride
            output[i, j] = np.max(array_padding[_i:_i+k_h,_j:_j+k_w])
    
    return output


# In[97]:


array = np.array([[i for i in range(1, 5)],
                  [i for i in range(6, 10)],
                  [i for i in range(11, 15)],
                  [i for i in range(16, 20)]])
kernel_shape = (2,2)
print(array)


max_pooling(array, kernel_shape, 2, 0)


# In[98]:


#平均プーリング
#ストライド，パディング指定可能

def mean_pooling(array, kernel_shape, stride, padding):
    a_h, a_w = array.shape
    k_h, k_w = kernel_shape
    row = int(((a_h +2*padding) - k_h)/stride)+1
    column = int(((a_w +2*padding) - k_w)/stride)+1
    output = np.zeros((row, column))
    
    #arrayにパディングを付ける
    array_padding = np.zeros((a_h+2*padding, a_w+2*padding))
    array_padding[padding:padding+a_h,padding:padding+a_w]=array
    print(array_padding)
    #畳み込みの計算
    for i in range(row):
        _i = i*stride
        for j in range(column):
            _j = j*stride
            output[i, j] = np.mean(array_padding[_i:_i+k_h,_j:_j+k_w])
    
    return output


# In[100]:


array = np.array([[i for i in range(1, 5)],
                  [i for i in range(6, 10)],
                  [i for i in range(11, 15)],
                  [i for i in range(16, 20)]])
kernel_shape = (2,2)
print(array)


mean_pooling(array, kernel_shape, 1, 0)


# In[ ]:


