#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[3]:


# Importing Dataset
dataPatient = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Patients- Selected Features1.xlsx')
dataNormal = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Absolutely Normal- Selected Features1.xlsx')
X = np.concatenate(  (dataPatient,dataNormal),axis=0  )

y = [0]*len(dataNormal)+[1]*len(dataPatient)


# In[4]:


# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)


# In[5]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


# In[6]:


tree.plot_tree(clf)
[...]


# In[7]:


# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth = 5, min_samples_leaf = 5, random_state = 100)


# In[8]:


# Performing training
clf_entropy.fit(X_train, y_train)


# In[9]:


# Creating the classifier object
clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth= 5, min_samples_leaf=5, random_state = 100)


# In[10]:


# Performing training
clf_gini.fit(X_train, y_train)


# In[11]:


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


# In[12]:


y_pred=clf_entropy.predict(X_test)


# In[13]:


cal_accuracy(y_test, y_pred)


# In[14]:


y_pred=clf_gini.predict(X_test)


# In[15]:


cal_accuracy(y_test, y_pred)

