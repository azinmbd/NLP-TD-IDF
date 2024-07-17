#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD


# In[3]:


docs = ['First try with SVD and CSR', 'First with Pipeline and second with Kmeans']
title = ['First doc', 'Second doc']


# In[4]:


tf1 = TfidfVectorizer()


# In[7]:


csr1 = tf1.fit_transform(docs)


# In[33]:


csr1


# In[8]:


print(csr1.toarray())


# In[10]:


print(csr1)


# In[12]:


words1 = tf1.get_feature_names_out()


# In[13]:


print(words1)


# In[34]:


print(tf1.get_stop_words())


# In[15]:


#counte Vectorizer


# In[16]:


vect1 = CountVectorizer()


# In[17]:


vect1.fit(docs)


# In[18]:


print("Vocabulary: ", vect1.vocabulary_)


# In[19]:


vector = vect1.transform(docs)


# In[21]:


print("Encoded Document is: ")
print(vector.toarray())


# In[22]:


doc2 = ['First try with SVD and CSR', 'First with Pipeline and second with Kmeans'
       'First with SVD' , 'Pipeline First']
title2 = ['First doc', 'secon doc', 'third doc', 'forth doc']


# In[23]:


csr2 = tf1.fit_transform(doc2)


# In[24]:


svd1 = TruncatedSVD(n_components=2)


# In[25]:


kmeans = KMeans(n_clusters=2)


# In[26]:


pipeline = make_pipeline(svd1, kmeans)


# In[27]:


pipeline.fit(csr2)


# In[28]:


labels2 = pipeline.predict(csr2)


# In[29]:


df= pd.DataFrame({'Lables': labels2, 'Docs':doc2})


# In[32]:


df1 = df.sort_values('Lables')
print(df1)


# In[ ]:




