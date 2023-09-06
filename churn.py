#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib as plt


# In[41]:



doc = pd.read_excel("C:/Users/STAFF/Desktop/Git_Repos/Telco_customer_churn.xlsx")
doc.sample(5)


# In[42]:


doc.drop(columns =["CustomerID","Count","Churn Reason","Churn Label","Churn Score","Lat Long"], inplace = True)


# In[43]:


doc.shape


# In[44]:


doc = doc.dropna()
doc.shape


# In[45]:


doc = doc[doc["Total Charges"] != " "]
doc.shape


# In[46]:


doc["Total Charges"] = pd.to_numeric(doc["Total Charges"])


# In[47]:


doc.dtypes


# In[48]:


def values_in_each_col(dataframe):
    for i in dataframe:
        if dataframe[i].dtypes == "object":
            print(f"{i} : {dataframe[i].unique()}")


# In[49]:


values_in_each_col(doc)


# In[50]:


doc.replace("No internet service" , "No", inplace = True)
doc.replace("No phone service" , "No", inplace = True)


# In[51]:


values_in_each_col(doc)


# In[52]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in doc.columns:
    if doc[i].dtypes == "object":
         doc[i] = encoder.fit_transform(doc[i])

doc.sample(5)


# In[53]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# doc = minmax.fit_transform(doc.values)


# doc.iloc[:, :] = minmax.fit_transform(doc.values)

# doc = pd.DataFrame(doc) 
# doc.sample(5)


# In[54]:


import pickle

pickle.dump(scaler,open('scaling.pkl','wb'))


# In[55]:


import pandas as pd

doc = pd.DataFrame(doc) 

X = doc.drop(columns = "Churn Value", axis= "columns")
y= doc["Churn Value"]


# In[56]:


X.shape


# In[62]:


from sklearn.model_selection import train_test_split

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size= 0.70, random_state=None)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

X_train.shape


# In[63]:


scaler.fit(X_train)


# In[64]:


X_test = scaler.transform(X_test)


# In[65]:


import tensorflow


# In[66]:


import tensorflow
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(26, input_shape = (26,), activation = "relu"),
    keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(X_train, y_train, epochs = 100)


# In[67]:


y_pred = model.predict(X_test)


# In[68]:


model.evaluate(X_test, y_test)


# In[69]:


y_pred[:5]


# In[70]:


y_pred_final = []
for i in y_pred:
    if i > 0.5:
        y_pred_final.append(1)
    else:
        y_pred_final.append(0)


# In[71]:


y_pred_final[:12]


# In[72]:


from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred_final))


# In[73]:


import seaborn as sns
import matplotlib.pyplot as plt

cM =  tensorflow.math.confusion_matrix(labels = y_test, predictions= y_pred_final)

plt.figure(figsize = (8,5))
sns.heatmap(cM, annot =True, fmt ="d")
plt.xlabel('Predicted')
plt.ylabel('Real')


# In[74]:


###Pickle the file for deployment


# In[75]:


import pickle


# In[76]:


pickle.dump(model,open('churn_model.pkl','wb'))


# In[77]:


pickled_model = pickle.load(open('churn_model.pkl','rb'))


# In[ ]:





# In[ ]:




