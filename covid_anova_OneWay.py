#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries

import pandas as pd
import numpy as np


# In[2]:


import scipy.stats as stats
import os
import random


# In[3]:


import statsmodels.api as sm
import statsmodels.stats.multicomp


# In[4]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# In[5]:


os.getcwd()


# In[8]:


os.chdir('C:\\Users\\hp\\Desktop\\covid')


# In[9]:


os.getcwd()


# In[10]:


#Load data
StatewiseTestingDetails=pd.read_csv('./StatewiseTestingDetails.csv')
population_india_census2011=pd.read_csv('./population_india_census2011.csv')


# In[11]:


population_india_census2011.head()


# In[12]:


StatewiseTestingDetails.head()


# In[13]:


StatewiseTestingDetails['Positive'].sort_values().head()


# In[14]:


#list down the states which have 0 corona cases
StatewiseTestingDetails['State'][StatewiseTestingDetails['Positive']==0].unique()


# In[15]:


#List down the states which have 1 corona case
StatewiseTestingDetails['State'][StatewiseTestingDetails['Positive']==1].unique()


# In[16]:


#we see that there're many entries with 0. that meansno case has been detected, so we can add 1 in all entries.
#so while performing any sort of data transfromation that involves log in it, won't give error.
StatewiseTestingDetails['Positive']=StatewiseTestingDetails['Positive']+1 


# In[17]:


StatewiseTestingDetails['Positive'].sort_values()


# In[19]:


#Imput missing values by median of each state
stateMedianData=StatewiseTestingDetails.groupby('State')[['Positive']].median().                                    reset_index().rename(columns={'Positive':'Median'})
stateMedianData.head()


# In[20]:


StatewiseTestingDetails.head()


# In[21]:


for index,row in StatewiseTestingDetails.iterrows():
    if pd.isnull(row['Positive']):
        StatewiseTestingDetails['Positive'][index]=int(stateMedianData['Median'][stateMedianData['State']==row['State']])


# In[22]:


StatewiseTestingDetails['Positive'].sort_values()


# In[28]:


StatewiseTestingDetails.columns


# In[29]:


population_india_census2011.columns


# In[32]:


population_india_census2011.columns = ['Sno', 'State', 'Population', 'Rural population',
       'Urban population', 'Area', 'Density', 'Gender Ratio']


# In[33]:


population_india_census2011.columns


# In[34]:


#Merge StatewiseTestingDetails & population_india_census2011 dataframes
data=pd.merge(StatewiseTestingDetails,population_india_census2011,on='State')


# In[35]:


data.head()


# In[36]:


data.shape


# In[37]:


##Sort the Data Frame
data['Positive'].sort_values()


# In[38]:


#Write a function to create densityGroup bucket
def densityCheck(data):
    data['density_Group']=0
    for index,row in data.iterrows():
        status=None
        i=row['Density'].split('/')[0]
        try:
            if (',' in i):
                i=int(i.split(',')[0]+i.split(',')[1])
            elif ('.' in i):
                i=round(float(i))
            else:
                i=int(i)
        except ValueError as err:
            pass
        try:
            if (0<i<=300):
                status='Dense1'
            elif (300<i<=600):
                status='Dense2'
            elif (600<i<=900):
                status='Dense3'
            else:
                status='Dense4'
        except ValueError as err:
            pass
        data['density_Group'].iloc[index]=status
    return data


# In[39]:


data.columns


# In[40]:


data['Positive'].sort_values()


# In[41]:


#Map each state as per its density group
data=densityCheck(data)


# In[42]:


#We'll export this data so we can use it for Two - way ANOVA test.
stateDensity=data[['State','density_Group']].drop_duplicates().sort_values(by='State')


# In[44]:


data['Positive'].sort_values()


# In[45]:


data.to_csv('data.csv',index=False)
stateDensity.to_csv('stateDensity.csv',index=False)


# In[46]:


data.head()


# In[47]:


data.describe()


# In[48]:


#Rearrange dataframe

df=pd.DataFrame({'Dense1':data[data['density_Group']=='Dense1']['Positive'],
                 'Dense2':data[data['density_Group']=='Dense2']['Positive'],
                 'Dense3':data[data['density_Group']=='Dense3']['Positive'],
                 'Dense4':data[data['density_Group']=='Dense4']['Positive']})


# In[50]:


data.isna().sum()


# In[51]:


data[data['Positive'].isna()]


# In[52]:


df.dtypes


# In[53]:


####################### Approach 1.##########


# In[54]:


np.random.seed(1234)
dataNew=pd.DataFrame({'Dense1':random.sample(list(data['Positive'][data['density_Group']=='Dense1']), 10),
                      'Dense2':random.sample(list(data['Positive'][data['density_Group']=='Dense1']), 10),
                      'Dense3':random.sample(list(data['Positive'][data['density_Group']=='Dense1']), 10),
                      'Dense4':random.sample(list(data['Positive'][data['density_Group']=='Dense1']), 10)})


# In[55]:





# In[56]:


dataNew.head()


# In[57]:


dataNew.describe()


# In[58]:


dataNew['Dense1'].sort_values().head()


# In[59]:


dataNew.describe()


# In[60]:


dataNew['Dense1'].sort_values().head()


# In[62]:


#Plot number of Corona cases across different density groups to check their distribution.
fig = plt.figure(figsize=(10,10))
title = fig.suptitle("Corona cases across different density groups", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(2,2,1)
ax1.set_title("density Group-Dense1 & Corona Cases")
ax1.set_xlabel("density Group -Dense1")
ax1.set_ylabel("Corona Cases") 
sns.kdeplot(dataNew['Dense1'], ax=ax1, shade=True,bw=4, color='g')

ax2 = fig.add_subplot(2,2,2)
ax2.set_title("density Group -Dense2 & Corona Cases")
ax2.set_xlabel("density Group -Dense2")
ax2.set_ylabel("Corona Cases") 
sns.kdeplot(dataNew['Dense2'], ax=ax2, shade=True,bw=4, color='y')

ax2 = fig.add_subplot(2,2,3)
ax2.set_title("density Group -Dense2 & Corona Cases")
ax2.set_xlabel("density Group -Dense3")
ax2.set_ylabel("Corona Cases") 
sns.kdeplot(dataNew['Dense3'], ax=ax2, shade=True,bw=4, color='r')

ax2 = fig.add_subplot(2,2,4)
ax2.set_title("density Group -Dense4 & Corona Cases")
ax2.set_xlabel("density Group -Dense4")
ax2.set_ylabel("Corona Cases") 
sns.kdeplot(dataNew['Dense4'], ax=ax2, shade=True,bw=4, color='b')


# In[63]:


## Apply BoxCox Transformation to bring the data to close to Gaussian Distribution 
dataNew['Dense1'],fitted_lambda = stats.boxcox(dataNew['Dense1'])
dataNew['Dense2'],fitted_lambda = stats.boxcox(dataNew['Dense2'])
dataNew['Dense3'],fitted_lambda = stats.boxcox(dataNew['Dense3'])
dataNew['Dense4'],fitted_lambda = stats.boxcox(dataNew['Dense4'])


# In[64]:


#Apply log transformation to treat outliers and to bring to normal distribution


# In[65]:


dataNew.describe()


# In[66]:


dataNew.head()


# In[67]:


dataNew['Dense1'].describe()


# In[68]:


#Plot different density groups
fig = plt.figure(figsize=(10,10))
title = fig.suptitle("Corona cases across different density groups", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(2,2,1)
ax1.set_title("density Group-Dense1 & Corona Cases")
ax1.set_xlabel("density Group -Dense1")
ax1.set_ylabel("Corona Cases") 
sns.kdeplot(dataNew['Dense1'], ax=ax1, shade=True,bw=4, color='g')

ax2 = fig.add_subplot(2,2,2)
ax2.set_title("density Group -Dense2 & Corona Cases")
ax2.set_xlabel("density Group -Dense2")
ax2.set_ylabel("Corona Cases") 
sns.kdeplot(dataNew['Dense2'], ax=ax2, shade=True,bw=4, color='y')

ax2 = fig.add_subplot(2,2,3)
ax2.set_title("density Group -Dense2 & Corona Cases")
ax2.set_xlabel("density Group -Dense3")
ax2.set_ylabel("Corona Cases") 
sns.kdeplot(dataNew['Dense3'], ax=ax2, shade=True,bw=4, color='r')

ax2 = fig.add_subplot(2,2,4)
ax2.set_title("density Group -Dense4 & Corona Cases")
ax2.set_xlabel("density Group -Dense4")
ax2.set_ylabel("Corona Cases") 
sns.kdeplot(dataNew['Dense4'], ax=ax2, shade=True,bw=4, color='b')


# In[69]:


#############Assumptions check - Normality
stats.shapiro(dataNew['Dense1'])


# In[70]:


stats.shapiro(dataNew['Dense2'])


# In[71]:


stats.shapiro(dataNew['Dense3'])


# In[72]:


stats.shapiro(dataNew['Dense4'])


# In[73]:


# Levene variance test  
stats.levene(dataNew['Dense1'],dataNew['Dense2'],dataNew['Dense3'],dataNew['Dense4'])


# In[79]:


##p-value is more than 0.05 , So we can say that variances among groups are equal.


# In[74]:


F, p = stats.f_oneway(dataNew['Dense1'],dataNew['Dense2'],dataNew['Dense3'],dataNew['Dense4'])
# Seeing if the overall model is significant
print('F-Statistic=%.3f, p=%.3f' % (F, p))


# In[75]:


#Rearrange DataFrame
newDf=dataNew.stack().to_frame().reset_index().rename(columns={'level_1':'density_Group',
                                                               0:'Count'})
del newDf['level_0']


# In[76]:


# using Ols Model
model = ols('Count ~ C(density_Group)', newDf).fit()
model.summary()


# In[77]:


# Seeing if the overall model is significant
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")


# In[78]:


# Creates the ANOVA table
res = sm.stats.anova_lm(model, typ= 2)
res


# In[80]:


#So Based on p-value we can reject the H0; that is there's no significant difference as per density of an area 
#and number of corona cases


# In[81]:


#So what if you find statistical significance?  Multiple comparison tests

#When you conduct an ANOVA, you are attempting to determine if there is a statistically significant difference among the groups.
#If you find that there is a difference, you will then need to examine where the group differences lay.


# In[82]:


newDf.dtypes


# In[83]:


newDf.head()


# In[84]:


#Post hoc test
mc = statsmodels.stats.multicomp.MultiComparison(newDf['Count'],newDf['density_Group'])
mc_results = mc.tukeyhsd()
print(mc_results)


# In[85]:


#tuckey HSD test clearly says that there's a significant difference between Group1 & Group4


# In[86]:


#Above results from Tukey HSD suggests that except Dense1-Dense4 groups, all other pairwise comparisons for number of 
#corona cases rejects null hypothesis and indicates statistical significant differences.


# In[87]:


### Normality Assumption check
w, pvalue = stats.shapiro(model.resid)
print(w, pvalue)


# In[88]:


#Homogeneity of variances Assumption check
w, pvalue = stats.bartlett(newDf['Count'][newDf['density_Group']=='Dense1'], newDf['Count'][newDf['density_Group']=='Dense2']
                           , newDf['Count'][newDf['density_Group']=='Dense3'], newDf['Count'][newDf['density_Group']=='Dense4'])
print(w, pvalue)


# In[89]:


# Q-Q Plot for Normal Distribution check-
#Check the Normal distribution of residuals
res = model.resid 
fig = sm.qqplot(res, line='s')
plt.show()


# In[ ]:




