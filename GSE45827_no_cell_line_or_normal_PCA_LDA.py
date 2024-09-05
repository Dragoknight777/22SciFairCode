#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# In[2]:


data1 = pd.read_csv(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Breast_GSE26910.csv")
data2 = pd.read_csv(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Breast_GSE45827.csv")
data3 = pd.read_csv(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Breast_GSE7904.csv")


# In[3]:


full_data = pd.concat([data2.iloc[:71,:], data2.iloc[92:,:]],join="inner",axis=0)


# In[4]:


X = full_data.iloc[:,2:]


# In[5]:


y = full_data.loc[:,["type"]]
y = pd.factorize(y.to_numpy().flatten())
target_names = y[1]
y = y[0]


# In[6]:


target_names[2] = 'luminal A'
target_names[3] = 'luminal B'
target_names


# In[7]:


X = X.to_numpy()


# In[8]:


pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=3)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first 3 components): %s'
      % str(pca.explained_variance_ratio_))


# In[9]:


plt.figure()
colors = ['navy', 'turquoise', 'green', 'red']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of GSE45827')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.savefig(r"C:\Users\dlin0\Documents\21-22_Science_Fair\1-25-22_GSE45827_PCA_figure")

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2, 3], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of GSE45827')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.savefig(r"C:\Users\dlin0\Documents\21-22_Science_Fair\1-25-22_GSE45827_LDA_figure")

