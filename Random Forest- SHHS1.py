#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the required packages
import numpy as np
import pandas as pd


# In[2]:


# Importing Dataset
dataPatient = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Stroke\Patients- Stroke- SF1.xlsx')
dataNormal = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Stroke\Absolutely Normal- SF1.xlsx')
X = np.concatenate(  (dataPatient,dataNormal), axis=0  )

y = [0]*len(dataNormal)+[1]*len(dataPatient)


# In[3]:


len(X)


# In[4]:


# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# In[20]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 10, max_features="auto", random_state = 44)
rf_model.fit(X_train, y_train)


# In[21]:


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


# In[22]:


y_pred = rf_model.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[ ]:





# In[ ]:




