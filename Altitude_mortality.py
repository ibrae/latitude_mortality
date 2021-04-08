#!/usr/bin/env python
# coding: utf-8

# In[120]:


# import libraries
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[121]:


#1: Load the data and print the column names
df= pd.read_csv("lmdata.csv")
df


# In[122]:


##2: Generate descriptive statistics for the data
df.describe(include = "all").astype(int)


# In[123]:


##3: Create a line plot for the variables. Add a title and x & y axes
sns.lineplot('latitude', 'mortality',data=df, color='blue')
plt.xlabel('Latitude')
plt.ylabel('Mortality')
plt.title("Motalitity to latitude on a lineplot", color="red")
plt.show()


# In[124]:


#Create a scatter plot / Add title and (X,Y) axis names
sns.scatterplot('latitude','mortality', data=df, color='blue')
plt.xlabel('latitude',fontsize=14)
plt.ylabel('Mortality', fontsize=14)
plt.title("latitude vs mortality on scatterplot", color='red', fontsize=17)
plt.show()


# In[125]:


# Create a boxplot for mortality
sns.boxplot('mortality',data=df, width=0.5,palette="colorblind")
plt.title('mortality boxplot.')
plt.show()


# In[126]:


#Conduct a Pearson’s correlation test for the variables
from scipy.stats import pearsonr
# Convert dataframe into series
list1 = df['latitude']
list2 = df['mortality']
  
# Apply the pearsonr
corr = pearsonr(list1, list2)
print('Pearsons correlation is: ', corr)


# In[127]:


#Create a pair plot for the data
sns.pairplot(df)
plt.title('latitude to mortality pairplot.')
plt.show()


# In[128]:


# Create a Seaborn regplot of the regression model and a 95% confidence interval
sns.regplot(x='latitude', y='mortality', data=df, color='g', marker='+')
plt.title('latitude to mortality replot. ')
plt.show()


# In[140]:


#Create the regression model using ols() from the statsmodel package
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = sm.OLS.from_formula('mortality ~ latitude', data=df).fit()
# Print the model Summary
print(model.summary())

