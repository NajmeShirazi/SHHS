#!/usr/bin/env python
# coding: utf-8

# In[240]:


# Importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[241]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[242]:


# Importing Dataset
dataPatient = pd.read_excel(r'C:\Users\Najmeh\Desktop\New Patient\patients-15feature.xlsx')
dataNormal = pd.read_excel(r'C:\Users\Najmeh\Desktop\New Patient\Absolutely Normal- 15 Feature.xlsx')
X = np.concatenate(  (dataPatient,dataNormal),axis=0  )

y = [0]*len(dataNormal)+[1]*len(dataPatient)


# In[243]:


# Splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)


# In[244]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


# In[245]:


tree.plot_tree(clf)
[...]


# In[246]:


# Decision tree with entropy
tree = DecisionTreeClassifier(criterion='entropy', max_depth = 5, min_samples_leaf = 5, random_state = 100)


# In[247]:


# Performing training
tree.fit(X_train, y_train)


# In[248]:


X_combined = np.vstack((X_train, X_train))
y_combined = np.hstack((y_train, y_test))


# In[249]:


# Creating the classifier object
clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth= 5, min_samples_leaf=5, random_state = 100)


# In[250]:


# Performing training
clf_gini.fit(X_train, y_train)


# In[251]:


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


# In[252]:


y_pred=tree.predict(X_test)


# In[253]:


cal_accuracy(y_test, y_pred)


# In[254]:


y_pred=clf_gini.predict(X_test)


# In[255]:


cal_accuracy(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




