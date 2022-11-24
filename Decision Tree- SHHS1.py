#!/usr/bin/env python
# coding: utf-8

# In[22]:


# Importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[23]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[24]:


# Importing Dataset
dataPatient = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Stroke\Patients- Stroke- SF1.xlsx')
dataNormal = pd.read_excel(r'C:\Users\Najmeh\Desktop\New\Stroke\Absolutely Normal- SF1.xlsx')
X = np.concatenate(  (dataPatient,dataNormal), axis=0  )

y = [0]*len(dataNormal)+[1]*len(dataPatient)


# In[25]:


# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)


# In[26]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)


# In[27]:


tree.plot_tree(clf)
[...]


# In[36]:


# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth = 3, min_samples_leaf = 3, random_state = 100)


# In[37]:


# Performing training
clf_entropy.fit(X_train, y_train)


# In[38]:


# Creating the classifier object
clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth= 3, min_samples_leaf= 3, random_state = 100)


# In[39]:


# Performing training
clf_gini.fit(X_train, y_train)


# In[40]:


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


# In[41]:


y_pred=clf.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[42]:


y_pred=clf_entropy.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[43]:


y_pred=clf_gini.predict(X_test)
cal_accuracy(y_test, y_pred)


# In[ ]:





# In[ ]:




