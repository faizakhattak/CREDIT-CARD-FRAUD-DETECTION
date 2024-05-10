#!/usr/bin/env python
# coding: utf-8

# # CREDIT CARD FRAUD DETECTION

# ## Import Dataset 

# In[1]:


import pandas as pd
df =pd.read_csv('creditcard.csv')
df


# ## Data Preprocessing 

# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


print(f"Number of columns: {df.shape[1]}")
print(f"Number of rows: {df.shape[0]}")


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().any()


# In[9]:


df = df.drop_duplicates()


# In[10]:


df.shape


# ## normalization using the StandardScaler

# In[11]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create a pipeline for normalization
scaler = make_pipeline(StandardScaler())

# Normalize numerical features (columns V1 to V28)
df_copy = df.copy()
df_copy.loc[:, 'V1':'V28'] = scaler.fit_transform(df_copy.loc[:, 'V1':'V28'])



# In[12]:


from imblearn.over_sampling import RandomOverSampler
# Apply oversampling to handle class imbalance
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(df.drop("Class", axis=1), df["Class"])




# In[13]:


df['Class'].value_counts()


# ## visualization

# In[14]:


import matplotlib.pyplot as plt

plt.style.use('ggplot')
df['Class'].value_counts().plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Class')
plt.show()


# ## Training and Testing

# In[15]:


from sklearn.model_selection import train_test_split
## Split the resampled dataset into features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[16]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


# In[17]:


#Make predictions on the test data
y_pred = rf_classifier.predict(X_test)


# In[18]:


#Evaluate the model
print(classification_report(y_test, y_pred))


# In[20]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\n Accuracy: {accuracy}")
print(f" Precision: {precision}")
print(f" Recall: {recall}")
print(f" F1 Score: {f1}")
    


# ### An accuracy of approximately 0.99997 means that the model is correct in predicting nearly all instances in the dataset.

# ### A precision of approximately 0.99995 means that almost all transactions predicted as fraudulent by the model are indeed fraudulent.
# 

# ### A recall of 1.0 indicates that the model correctly identifies all fraudulent transactions in the dataset.
# An F1 Score of approximately 0.99997 suggests that the model performs very well in terms of both precision and recall.
# The model has achieved high accuracy, precision, recall, and F1 Score, indicating it performs very well in detecting fraudulent
# credit card transactions. However, it's important to remember that there could still be false positives and false negatives. 
# It's crucial to consider the real-world impact of these predictions

# In[ ]:




