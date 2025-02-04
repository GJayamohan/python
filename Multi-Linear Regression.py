#!/usr/bin/env python
# coding: utf-8

# #### Assumptions in Multilinear regression
# - Linearity: The relationship between the predictors and the response is linear.
# - Independence: Observations are independent of each other.
# - Homoscedasticity: The residuals (difference between observed and predicted values) exhibit constant variance at all levels of the predictor.
# - Normal Distribution of Errors: The residuals of the model are normally distributed.
# - No multicillinearity: The independent variables should not be too highly correlated with each other.

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[5]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[6]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# #### Description of columns 
# - MPG : Milege of the car(Mile per Galllon)
# + HP : Horse Power of the car
# - vOL : volume of the car(sixe)
# - SP : Top speed 0f the car (Miler per hour)
# - WT : weight of the car(pounds)

# In[7]:


cars.info()


# In[8]:


cars.isna().sum()


# #### Observations
# - There are no missing values
# - There are 81 observations
# - The data types of the columns are relevant abd valid

# In[11]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()

