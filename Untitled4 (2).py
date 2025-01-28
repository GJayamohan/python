#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[9]:


print(type(data))
print(data.shape)
print(data.size)


# In[10]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[11]:


data1.info()


# In[12]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[13]:


data1[data1.duplicated()]


# In[14]:


data1[data1.duplicated(keep = False)]


# In[15]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[16]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[17]:


data.isnull().sum()


# In[18]:


cols = data1.columns
colors = ['yellow', 'maroon',]
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[19]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[20]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[21]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ", median_solar)
print("Mean of Solar: ", mean_solar)


# In[22]:


data1['Solar'] = data1['Solar'].fillna(median_ozone)
data1.isnull().sum()


# In[23]:


data1.head()


# In[24]:


print(data1["Weather"].value_counts())
mode_weather  =data1["Weather"].mode()[0]
print(mode_weather)


# In[25]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[26]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[28]:


data1["Ozone"].describe()


# In[33]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# In[35]:


import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# Observations from Q-Q plot
#     The data does not follow normal distribution as the data are deviating signifiantly away from red line
#     The data shows a right skewed distribution and possible outliers
