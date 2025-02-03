#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[58]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[59]:


data1.info()


# In[60]:


data1.describe()


# In[61]:


plt.scatter(data1["daily"],data1["sunday"])


# In[62]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[63]:


model.summary()


# In[64]:


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

# In[65]:


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

# In[66]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[67]:


model1.summary()


# #### Observations
# - The predicted equation is y_hat=beta_0 + beta_1x
# - beta_0 = 13.8356, beta_1 = 1.3397x

# - The probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# - Therefore the intercept coefficent may not be that much significant in prediction 
# - However in p-value for "daily" (beta_1) is 0.00 < 0.05
# - Therfore the beta_1 coefficent is highly significant and is contibuting to prediction.

# #### Fit a Linear Regression Model

# In[68]:


# Build regression model
import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[69]:


model1.summary()


# In[70]:


model.params


# In[76]:


print(f'model1 t-values:\n{model1.tvalues}\n----------------\nmodel p-values: \n{model.pvalues}')


# In[85]:


(model1.rsquared,model1.rsquared_adj)


# In[87]:


newdata=pd.Series([200,300,1500])


# In[88]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[89]:


model1.predict(data_pred)


# In[90]:


# Predict on all given training data
pred = model1.predict(data1["daily"])
pred


# In[91]:


# Add predicted values as a column in data1
data1["Y_hat"] = pred
data1


# In[94]:


data1["residuals"]=data1["sunday"]-data1["Y_hat"]
data1


# In[95]:


# Compute Mean Squared Error for the model
mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# #### Checking the model rediuals scatter plot

# In[96]:


plt.scatter(data1["Y_hat"], data1["residuals"])


# #### Observations
# - The residuals data points are randomly scatterd
# - There appears to be no trend and the residuals are randomly placed around the zero error line
# - Hence the assumption of homoscedasticty is satisfed(constant variance in residuals)

# In[97]:


import statsmodels.api as sm 
sm.qqplot(data1["residuals"], line='45', fit=True)
plt.show(0)


# In[ ]:




