#!/usr/bin/env python
# coding: utf-8

# In[192]:


import pandas as pd
import numpy as np
from sklearn import linear_model
# import matplotlib.pyplot as plt
from sklearn.svm import SVC


# In[193]:


# Importing Dataset
dataPatient = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Patients- Selected Features1.xlsx')
dataNormal = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Absolutely Normal- Selected Features1.xlsx')
X = np.concatenate(  (dataPatient,dataNormal),axis=0  )

y = [0]*len(dataNormal)+[1]*len(dataPatient)


# In[194]:


len(X)


# In[195]:


from sklearn.model_selection import train_test_split


# In[196]:


# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[197]:


from sklearn.svm import SVC
#linear_model = SVC(kernel='linear', C = 10)
#linear_model.fit(X, y)


# In[198]:


rbf_model = SVC(kernel='rbf', gamma = 0.7, C = 1)
rbf_model.fit(X, y)


# In[199]:


poly_model = SVC(kernel='poly', degree = 7, C = 15)
poly_model.fit(X, y)


# In[200]:


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


# In[201]:


y_pred = rbf_model.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[202]:


y_pred = poly_model.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[203]:


#y_pred = linear_model.predict(X_test)
#cal_accuracy(y_test, y_pred)


# In[ ]:





# In[ ]:




