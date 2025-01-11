#!/usr/bin/env python
# coding: utf-8

# #Example
# 1.Read a long text in to string that contains some punctuation marks also.
# 2.Convert to lower case.
# 3.Remove ounctuation marks.
# 4.Split in to words using split() function.
# 5.Make a dictionary with each unique word 
# 6. Make a data frame.

# In[3]:


import pandas as pd
import string
text = "The preliminary analysis of the data retrieved from the FDR (flight data recorder) and CVR (cockpit voice recorder) of the ill-fated Coast Guard chopper has shown that the pilots “lost control three to four seconds” before the crash, an official told TOI."
print(text)


# In[4]:


text = text.lower()
text


# In[5]:


words = text.split()
words


# In[6]:


word_counts = {}
for word in words:
    word_counts[word] = word_counts.get(word, 0) + 1
word_counts


# In[ ]:




