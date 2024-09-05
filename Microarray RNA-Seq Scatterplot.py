#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt


# In[2]:


plt.style.use('seaborn')


# In[3]:


with open(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Results\8-20-22_Microarray_log2Fold.txt") as f:
    mArrData = f.readlines()
with open(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Results\8-20-22_RNA-seq_log2Fold.txt") as f:
    RNASeqData = f.readlines()


# In[4]:


for i in range(len(mArrData)):
    mArrData[i] = mArrData[i].rstrip()
    splitList = mArrData[i].split(", ")
    mArrData[i] = [splitList[0], splitList[1]]
for i in range(len(RNASeqData)):
    RNASeqData[i] = RNASeqData[i].rstrip()
    splitList = RNASeqData[i].split(", ")
    RNASeqData[i] = [splitList[0], splitList[1]]


# In[5]:


print(len(RNASeqData))
print(len(mArrData))


# In[6]:


for i in range(len(RNASeqData)):
    RNASeqData[i][1] = float(RNASeqData[i][1])
for i in range(len(mArrData)):
    mArrData[i][1] = float(mArrData[i][1])


# In[13]:


mArrFold = []
RNASeqFold = []
colors = []
for j in range(len(RNASeqData)):
    for i in range(len(mArrData)):
        if mArrData[i][0] == RNASeqData[j][0] and RNASeqData[j][1] != -1000000000:
            mArrFold.append(mArrData[i][1])
            RNASeqFold.append(RNASeqData[j][1])
            if RNASeqData[j][1] <= 0:
                colors.append('b')
            else:
                colors.append('r')
            break


# In[14]:


print(len(mArrFold))
print(len(RNASeqFold))


# In[15]:


coef = np.polyfit(mArrFold, RNASeqFold, 1)
poly1d_fn = np.poly1d(coef)


# In[16]:


poly1d_fn.c


# In[21]:


from sklearn.metrics import r2_score
R2 = r2_score(RNASeqFold, poly1d_fn(mArrFold))
print(math.sqrt(R2))
print(R2)


# In[20]:


#fig, ax = plt.subplots(figsize = (9, 9))

plt.scatter(mArrFold, RNASeqFold, c = colors, linewidth = 1, alpha = 0.75)

plt.plot(mArrFold, poly1d_fn(mArrFold), '--k')

plt.title('Microarray against RNA-Seq log2 Fold Changes')
plt.xlabel('Microarray log2 Fold Changes')
plt.ylabel('RNA-Seq log2 Fold Changes')

plt.tight_layout()

plt.savefig(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Results\Log2FoldChanges.png")


# In[19]:


RNASeqData


# In[ ]:




