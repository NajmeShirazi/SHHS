#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
from sklearn import linear_model
# import matplotlib.pyplot as plt
from sklearn.svm import SVC


# In[24]:


# Importing Dataset
dataPatient = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Stroke\Patients- Stroke- SF1.xlsx')
dataNormal = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Stroke\Absolutely Normal- SF1.xlsx')
X = np.concatenate(  (dataPatient,dataNormal), axis=0  )

y = [0]*len(dataNormal)+[1]*len(dataPatient)


# In[25]:


len(X)


# In[26]:


# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# In[27]:


from sklearn.svm import SVC
linear_model = SVC(kernel='linear', C = 15)
linear_model.fit(X_train, y_train)


# In[28]:


poly_model = SVC(kernel='poly', degree = 5, C = 15)
poly_model.fit(X_train, y_train)


# In[29]:


rbf_model = SVC(kernel='rbf', gamma = 0.3, C = 10)
rbf_model.fit(X_train, y_train)


# In[30]:


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


# In[31]:


y_pred = linear_model.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[32]:


y_pred = poly_model.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[33]:


y_pred = rbf_model.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[ ]:




