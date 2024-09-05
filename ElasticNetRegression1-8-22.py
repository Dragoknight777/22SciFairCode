#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from combat.pycombat import pycombat
from sklearn.linear_model import ElasticNetCV


# In[2]:


data1 = pd.read_csv(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Breast_GSE26910.csv")
data2 = pd.read_csv(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Breast_GSE45827.csv")
data3 = pd.read_csv(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Breast_GSE7904.csv")


# In[3]:


full_data = pd.concat([data1, data2.iloc[:71,:], data2.iloc[85:,:], data3],join="inner",axis=0)
batch = []
datasets = [data1,pd.concat([data2.iloc[:71,:],data2.iloc[85:,:]],join="inner",axis=0),data3]
for j in range(len(datasets)):
    batch.extend([j for _ in range(datasets[j].shape[0])])

# run pyComBat
df_corrected = pycombat(full_data.iloc[:,2:].transpose(),batch).transpose()


# In[4]:


headers = full_data.columns.drop(['samples', 'type']).to_numpy().flatten()
print(headers)
normal = df_corrected.iloc[83:90,:]
cancer_all = pd.concat([df_corrected.iloc[6:83,:],df_corrected.iloc[90:187,:]],join="inner",axis=0)


# In[5]:


print(normal)
print(type(normal))


# In[6]:


def importantGeneIdentification(iterations, cancerSamples):
    geneList = []
    geneAppearances = []
    r2Scores = []
    y = []
    for i in range(7):
        y.append(0)
    for i in range(cancerSamples):
        y.append(1)
    y = np.asarray(y) 
    for j in range(iterations):
        print(j)
        cancer = cancer_all.sample(n = cancerSamples)
        X = pd.concat([normal, cancer],join="inner",axis=0)
        # define the model
        model = ElasticNetCV(cv=5, random_state=0)
        # fit the model
        model.fit(X, y)
        r2Scores.append(model.score(X, y))
        # get importance
        importance = model.coef_
        import_gene = headers[importance != 0]
        for a in import_gene:
            if(geneList.count(a) > 0):
                geneAppearances[list(np.where(np.asarray(geneList) == a))[0][0]] += 1
            else:
                geneList.append(a)
                geneAppearances.append(1)
    lenOfList = len(geneList)
    print(str(geneList))
    print(str(geneAppearances))
    print(r2Scores)
    with open(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Results\4-24-22_100_iter_microarray_results.txt", 'a') as f:
       f.write(str(geneList))
       f.write('\n')
       f.write(str(geneAppearances))
       f.write('\n')
    for k in range(lenOfList):
       largestIndex = 0
       for b in range(len(geneList)):
           if(geneAppearances[b] > geneAppearances[largestIndex]):
               largestIndex = b
       print(geneList[largestIndex] + " num of appearances: " + str(geneAppearances[largestIndex]))
       with open(r"C:\Users\dlin0\Documents\21-22_Science_Fair\Results\4-24-22_100_iter_microarray_results.txt", 'a') as f:
           f.write(geneList[largestIndex] + " num of appearances: " + str(geneAppearances[largestIndex]))
           f.write('\n')
       del geneList[largestIndex]
       del geneAppearances[largestIndex]


# In[7]:


importantGeneIdentification(100, 20)


# In[ ]:





# In[ ]:




