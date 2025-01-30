#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[7]:


print(type(data))
print(data.shape)
print(data.size)


# In[8]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[9]:


data1.info()


# In[10]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[11]:


data1[data1.duplicated()]


# In[12]:


data1[data1.duplicated(keep = False)]


# In[13]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[14]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[15]:


data.isnull().sum()


# In[16]:


cols = data1.columns
colors = ['yellow', 'maroon',]
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[17]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[18]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[19]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ", median_solar)
print("Mean of Solar: ", mean_solar)


# In[20]:


data1['Solar'] = data1['Solar'].fillna(median_ozone)
data1.isnull().sum()


# In[21]:


data1.head()


# In[22]:


print(data1["Weather"].value_counts())
mode_weather  =data1["Weather"].mode()[0]
print(mode_weather)


# In[23]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[24]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[25]:


data1["Ozone"].describe()


# In[26]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]
for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# In[27]:


import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# #### Observations from Q-Q plot
# - The data does not follow normal distribution as the data are deviating signifiantly away from red line
# - The data shows a right skewed distribution and possible outliers

# #### Observation 
# - The correlation between wind and temp is observed to be negatively correlated with mild strength

# In[28]:


data1.info()


# In[30]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[31]:


data1_numeric.corr()


# #### Observations
# - The highest correlation is observed between ozone and Temperature(0.597087)
# - The next higher correlation strength is observed between ozone and wind(-0.523738)
# - The next higher correlation strength is observed between wind and temp(-0.441228)
# - The least correlation strength is observed betweeen solar and wind(-0.055874)

# In[33]:


sns.pairplot(data1_numeric)


# #### Transformations

# In[35]:


data2=pd.get_dummies(data1,columns=['Month','Weather'])
data2


# In[ ]:




