#!/usr/bin/env python
# coding: utf-8

# In[144]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[145]:


dataPatient = pd.read_excel(r'C:\Users\Najmeh\Desktop\New Patient\patients-15feature.xlsx')
dataNormal = pd.read_excel(r'C:\Users\Najmeh\Desktop\New Patient\Absolutely Normal- 15 Feature.xlsx')
X = np.concatenate(  (dataPatient,dataNormal),axis=0  )

y = [0]*len(dataNormal)+[1]*len(dataPatient)


# In[146]:


len(X)


# In[147]:


from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set()


# In[148]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[149]:


knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean') 
# grid search hyper parametre (in sklearn)
knn.fit(X_train, y_train)


# In[150]:


y_pred = knn.predict(X_test)


# In[151]:


from sklearn import metrics
print("KNN model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)


# In[152]:


y_pred


# In[153]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=44)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
rf_model.fit(X_train, y_train)


# In[154]:


y_pred = rf_model.predict(X_test)
y_pred


# In[155]:


metrics.accuracy_score(y_test, y_pred)*100


# In[156]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X=X_train)
X_test = pca.transform(X_test)


# In[157]:


confusion_matrix(y_test, y_pred)


# In[ ]:




