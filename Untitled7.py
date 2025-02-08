#!/usr/bin/env python
# coding: utf-8

# #### Assumptions in Multilinear regression
# - Linearity: The relationship between the predictors and the response is linear.
# - Independence: Observations are independent of each other.
# - Homoscedasticity: The residuals (difference between observed and predicted values) exhibit constant variance at all levels of the predictor.
# - Normal Distribution of Errors: The residuals of the model are normally distributed.
# - No multicillinearity: The independent variables should not be too highly correlated with each other.

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[4]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[5]:



cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Description of columns
# - MPG : Milege of the car(Mile per Galllon)
# - HP : Horse Power of the car
# - vOL : volume of the car(sixe)
# - SP : Top speed 0f the car (Miler per hour)
# - WT : weight of the car(pounds)

# In[6]:


cars.info()


# In[7]:


cars.isna().sum()


# #### Observations
# - There are no missing values
# - There are 81 observations
# - The data types of the columns are relevant abd valid

# In[8]:



fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# Observations from boxplt and histograms
# There are some extreme values (outliers) observed in towards the right tail os SP and HP distribution.
# The VOL and WT columns, a few outliers are observed in both tails of their distributions.
# The extreme values of cars data may have come from the specially dsigned nature of cars.
# As this is multi-dimensional data, the outliers with respect to spatial dimension may have to be c0nsidered while building the regression model.

# In[9]:



cars[cars.duplicated()]


# In[10]:



sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[11]:



cars.corr()


# #### Observations from correlation plots and Coefficients
# - Between x and y all the variables are showing moderate to high correlation strenghts highest being between HP and MGP
# - Therefore this dataset for building a multiple linear regression model tp predict MPG
# - Among x columns (x1,x2,x3and x4) same very high correlaton strenghts are observed betweeen SP and HP , VOL vs WT.
# - The high correlation among x columns is not desirable as it might lead to multi collinearity problem

# In[12]:


model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model.summary()


# #### Observations from model summary
# - The R squared adjusted R squared values are good end about 75% of variability in Y is explained by Xcolunns.
# - The probability value with respect to F-static is close to zero indicating taht al or some of X columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue aming themselves which need to be further explained

# 
# Performance metrices for model1

# In[13]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[14]:


pred_y1 = model.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[15]:


from sklearn.metrics import mean_squared_error 
mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"]) 
print("MSE :", mse) 
print("RMSE :",np.sqrt(mse))


# #### Checking for multicollinearity among X-columns using VIF method

# In[16]:


cars.head()


# In[17]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[18]:


cars1=cars.drop("WT", axis=1)
cars1.head()


# In[23]:


import statsmodels.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[22]:


model2.summary()


# #### Performance metrics for model2

# In[27]:


df2=pd.DataFrame()
df2["actual_y2"]=cars["MPG"]
df2.head()


# In[28]:


pred_y2 = model.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[29]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### Observations from model2 summary()
# - The adjusted R-suared value improved slightly to 0.76
# - All the p=values for model parameters are less than 5% hence they are significant
# - Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG
# - There is no improvement in MSE value

# #### Identification of High Influence points(spatial outliners)

# In[30]:


cars1.shape


# In[31]:


k=3
n=81
leverage_cutoff=3*((k+1)/n)
leverage_cutoff


# In[32]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model,alpha=0.5)
y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')
plt.show()


# #### Observations
# - From the above plot, it is evident that data points 65,70,76,78,79,80 are the influencers
# - as their H Leverage values are higher and size is higher

# In[33]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[35]:


cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[36]:


cars2


# #### Build model3 on cars2 dataset

# In[37]:


model3=smf.ols('MPG~VOL+SP+HP',data=cars2).fit()


# In[38]:


model3.summary()


# #### Performance Metrics for Model3

# In[45]:


df3=pd.DataFrame()
df3["actual_y3"]=cars["MPG"]
df3.head()


# In[46]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"] = pred_y3
df3.head()


# In[63]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### Comparison of models
# 
# | Metric         | Model 1 | Model 2 | Model 3|
# |----------------|---------|---------|--------|
# | R-squared      | 0.771   | 0.770   | 0.885  |
# | Adj. R-squared | 0.758   | 0.761   | 0.880  |
# | MSE            | 18.89   | 18.91   | 8.68   |
# | RMSE           | 4.34    | 4.34    | 2.94   |
# 

# #### Check the validity of model assumptions for model3

# In[48]:


model3.resid


# In[49]:


model.fittedvalues


# In[58]:


import statsmodels.api as sm
qqplot=sm.qqplot(model3.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[51]:


sns.displot(model3.resid, kde = True)


# In[55]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[56]:


plt.figure(figsize=(6,4))
plt.scatter(get_standardized_values(model3.fittedvalues),
            get_standardized_values(model3.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()

