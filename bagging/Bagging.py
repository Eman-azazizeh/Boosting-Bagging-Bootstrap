#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[6]:


df = pd.read_csv("diabetes.csv")
df.head(5)


# In[7]:


df.isnull().sum()


# In[8]:


X = df.drop("Outcome",axis="columns")
y = df.Outcome


# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[11]:


X_scaled[:3]


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# In[16]:


from sklearn.tree import DecisionTreeClassifier


# In[17]:


from sklearn.model_selection import cross_val_score


# In[18]:


scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
scores


# In[20]:


scores.mean()


# In[21]:


from sklearn.ensemble import BaggingClassifier


# In[22]:


bag_model = BaggingClassifier(
base_estimator=DecisionTreeClassifier(), 
n_estimators=100, 
max_samples=0.8, 
bootstrap=True,
oob_score=True,
random_state=0
)


# In[23]:


bag_model.fit(X_train, y_train)


# In[24]:


bag_model.oob_score_


# In[25]:


bag_model.score(X_test, y_test)


# In[ ]:




