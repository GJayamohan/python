#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[11]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[12]:


data1.info()


# In[13]:


data1.describe()


# In[14]:


plt.scatter(data1["daily"],data1["sunday"])


# In[15]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[16]:


model.summary()


# In[17]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
y_hat = b0 + b1*x
plt.plot(x, y_hat, color = "g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# #### Observations
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the boxplot

# In[18]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# #### Observations
# - The relationship between x(daily) and y(sunday) is seen to be linear as seeen from scatter plot
# - The correlation is strong positive with Pearson's correlation coefficent of 0.958154

# ***Fit a Linear Regression Model***

# In[19]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[21]:


model1.summary()


# #### Observations
# - The predicted equation is y_hat=beta_0 + beta_1x
# - beta_0 = 13.8356, beta_1 = 1.3397x

# - The probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# - Therefore the intercept coefficent may not be that much significant in prediction 
# - However in p-value for "daily" (beta_1) is 0.00 < 0.05
# - Therfore the beta_1 coefficent is highly significant and is contibuting to prediction.

# In[ ]:




