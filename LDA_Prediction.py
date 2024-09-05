#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.decomposition import SparsePCA
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd


# In[2]:


full_data = pd.read_csv(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Breast_GSE45827.csv")
full_data


# In[21]:


X = full_data.iloc[:,2:]
y = full_data.loc[:,["type"]]
y = pd.factorize(y.to_numpy().flatten())
y = y[0]
headers = full_data.columns.drop(['samples', 'type']).to_numpy().flatten()


# In[22]:


LDA_transformer = LinearDiscriminantAnalysis(n_components=5)
LDA_transformer.fit(X, y)


# In[23]:


LDA_transformer.coef_[0]


# In[15]:


# plt.plot(headers, LDA_transformer.coef_[0], label='component 0')
# #plt.plot(headers, LDA_transformer.coef_[1], label='component 1')
# # plt.plot(headers, LDA_transformer.components_[2], label='component 2')
# # plt.plot(headers, LDA_transformer.components_[3], label='component 3')
# # plt.plot(headers, LDA_transformer.components_[4], label='component 4')
# plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
# plt.title('LDA')
# plt.show()


# In[16]:


index = np.where(np.abs(LDA_transformer.coef_[0])>0.005)


# In[17]:


X = np.transpose(X)
X


# In[18]:


X = X.iloc[index]
X


# In[19]:


X = np.transpose(X)


# In[20]:


correct = 0
for i in range(0, 151):
    X_train = np.concatenate((X.iloc[:i,:], X.iloc[i+1:,:]))
    X_test = X.iloc[i:i+1,:]
    y_train = np.concatenate((y[:i], y[i+1:]))
    y_test = y[i:i+1]
    LDA_transformer_2 = LinearDiscriminantAnalysis(n_components=5)
    LDA_transformer_2.fit(X_train, y_train)
    y_predict = LDA_transformer_2.predict(X_test)
    #print(y_predict, y_test)
    if y_predict == y_test:
        correct += 1
print(X.shape[1])


# In[25]:


numGenes = []
accuracy = []
headers = full_data.columns.drop(['samples', 'type']).to_numpy().flatten()
for j in range(101):
    print(j)
    X = full_data.iloc[:,2:]
    index = np.where(np.abs(LDA_transformer.coef_[0])>(j*0.0001))
    X = np.transpose(X)
    X = X.iloc[index]
    X = np.transpose(X)
    correct = 0
    for i in range(0, 151):
        X_train = np.concatenate((X.iloc[:i,:], X.iloc[i+1:,:]))
        X_test = X.iloc[i:i+1,:]
        y_train = np.concatenate((y[:i], y[i+1:]))
        y_test = y[i:i+1]
        LDA_transformer_2 = LinearDiscriminantAnalysis(n_components=5)
        LDA_transformer_2.fit(X_train, y_train)
        y_predict = LDA_transformer_2.predict(X_test)
        #print(y_predict, y_test)
        if y_predict == y_test:
            correct += 1
    numGenes.append(X.shape[1])
    accuracy.append(correct/151)


# In[28]:


plt.plot(numGenes, accuracy)


# In[10]:


accuracy


# In[29]:


numGenes


# In[ ]:




