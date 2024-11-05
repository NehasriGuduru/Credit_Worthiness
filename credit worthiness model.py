#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# In[2]:


# Load data
data = pd.read_csv('credit_train.csv')


# In[16]:


data.head(10)


# In[3]:


# Preprocess data
X = data.drop('Credit_Score', axis=1) 
y = data['Credit_Score']   


# In[4]:


# Handle imbalanced data with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)


# In[5]:


# Scale the features
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)


# In[6]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, test_size=0.2, random_state=42)


# In[7]:


# Initialize Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)


# In[8]:


# Train the model
rf.fit(X_train, y_train)


# In[9]:


# Predict on test data
y_pred = rf.predict(X_test)


# In[11]:


#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[12]:


print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
#heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# In[ ]:




