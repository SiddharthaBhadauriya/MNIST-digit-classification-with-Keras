#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense


# In[3]:


(X_train, y_train),(X_test, y_test)= keras.datasets.mnist.load_data()


# In[4]:


print(X_train.shape)
print(y_train.shape)


# In[5]:


X_train[0]


# In[6]:


plt.imshow(X_train[0])
print(y_train[0])


# In[7]:


X_train = X_train/255
X_test = X_test/255
X_train[0]


# In[8]:


model = Sequential()


# In[9]:


model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))


# In[10]:


model.summary()


# In[19]:


model.compile(optimizer='Adam', loss= 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[27]:


history = model.fit(X_train, y_train, batch_size=32, epochs=15, verbose=1,validation_split = 0.2)


# In[29]:


model.evaluate(X_test, y_test)


# In[30]:


model.predict_classes(X_test)


# In[31]:


y_test[0]


# In[32]:


y_test[1]


# In[33]:


y_test[2]


# In[44]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


# In[48]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])


# In[ ]:




