#!/usr/bin/env python
# coding: utf-8

# In[73]:


# Importing the required packages
import numpy as np
import pandas as pd


# In[74]:


# Importing Dataset
dataPatient = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Patients- Selected Features1.xlsx')
dataNormal = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Absolutely Normal- Selected Features1.xlsx')
X = np.concatenate(  (dataPatient,dataNormal),axis=0  )

y = [0]*len(dataNormal)+[1]*len(dataPatient)


# In[75]:


len(X)


# In[76]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[77]:


# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# In[78]:


knn = KNeighborsClassifier(n_neighbors = 5, metric='euclidean') 
# grid search hyper parametre (in sklearn)
knn.fit(X_train, y_train)


# In[79]:


# Function to calculate accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))


# In[80]:


y_pred = knn.predict(X_test)
cal_accuracy(y_test, y_pred)

