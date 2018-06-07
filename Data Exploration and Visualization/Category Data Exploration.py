#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:06:22 2018

@author: sulbha
"""
##############################################################################
# 1. import statements
import pandas as pd
import matplotlib.pyplot as plt

##############################################################################


##############################################################################
# 2. load dataset and column names

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', \
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', \
        'hours-per-week', 'native-country', 'sal_class']

cols = [c.replace(' ', '-') for c in cols]

data = pd.read_csv(url, header = None)

data.columns = cols
##############################################################################


##############################################################################
# understand the data

print(data.head())
print(data.describe())
print(data.dtypes)
print(data.info())
##############################################################################

##############################################################################
# 3. Decoding Categorical data
# here column education-num is coded in form of continuous variable. So lets 
# decode it with understandable categories names.

print ("Unique values for education: ", data['education'].unique() )
print ("Unqiue values for education-num: ", data['education-num'].unique() )
edu_cat = data['education'].unique()
edu_code = data['education-num'].unique()
edu_dict = dict(zip(edu_code, edu_cat))

for k,v in edu_dict.items():
    replace = data.loc[:,'education-num']==k
    data.loc[replace, 'education-decode'] = v

print(data.head())

# plot original education category from the imported data
plt.figure(figsize=(10,10) )
plt.xticks( rotation = 45)
df = data.loc[:,'education'].value_counts().head(5)
print(df)
plt.hist(df, color='r' ,bins = 30)
plt.show()

# plot the decoded data
#plt.figure(figsize=(10,10) )
#plt.xticks( rotation = 45)
#plt.hist(data.loc[:,'education-decode'], bins = 30)
#plt.show()
##############################################################################


##############################################################################
# 4. Imputing missing values
print(data.loc[:,'education-decode'].value_counts())

print(data.loc[:,'education-num'].value_counts())

print(data.loc[:,'relationship'].value_counts())
## - since there are no missing values so no imputation is needed
####################################################

##############################################################################

# 5. consolidating categories
# mark ' Some-college' as 'HS-grad' as they are likely to belong to high schoolers to graduation category 
# there is leading space in this category

# replotting the graph
#plt.figure(figsize=(10,10) )
#plt.xticks( rotation = 45)
#plt.hist(data.loc[:,'education-decode'], bins = 30)
#plt.show()

# start consoidation based on education levels
lst = [' 10th',' 11th', ' 12th', ' Prof-school',' Some-college', ' HS-grad', ' Bachelors', ' Assoc-voc', ' Assoc-acdm']
for l in lst:
    data.loc[data.loc[:,'education-decode']==l, 'education-decode'] = 'HS-grad'

lst = [' 7th-8th', ' 9th', ' 5th-6th', ' 1st-4th', ' Preschool']
for l in lst:
    data.loc[data.loc[:,'education-decode']==l, 'education-decode'] = 'under-grad'

data.loc[data.loc[:,'education-decode']==' Masters', 'education-decode'] = 'over-grad'
data.loc[data.loc[:,'education-decode']==' Doctorate', 'education-decode'] = 'over-grad'

# replotting the graph after consolidation
#plt.figure(figsize=(10,10) )
#plt.xticks( rotation = 45)
#plt.hist(data.loc[:,'education-decode'], bins = 30)
#plt.show()

# check value counts
data['education-decode'].value_counts()

##############################################################################
#6. one hot encoding for 'education-decode' as it has 3 categores. Adding three columns for under-grad, HS-grad, over-grad
data.loc[:, 'under-grad'] = (data.loc[:, 'education-decode']=='under-grad').astype(int)
data.loc[:, 'over-grad'] = (data.loc[:, 'education-decode']=='over-grad').astype(int)
data.loc[:, 'HS-grad'] = (data.loc[:, 'education-decode']=='HS-grad').astype(int)

print(data.head())
##############################################################################

##############################################################################
#7. drop column 'education-decode' as it is no londer needed.
data.drop('education-decode',axis = 1 ,inplace = True)
print(data.head())
##############################################################################

##############################################################################
#8.  plot categories

plt.hist(data['under-grad'])
plt.show()

sel_lst =['sex', 'under-grad', 'HS-grad', 'over-grad']
df_sex = data.loc[:,sel_lst]
print(df_sex.head())

# exploring distribution by sex
df_sex = df_sex.groupby('sex')

df_sex.hist()
plt.show()

##############################################################################

#10 - summary
# Education category was decoded from numerial codes to descriptive codes
# using subject matter knowledge e.g. all adults with edicaion level as < 9 were
# categorized as 'under-grad' etc. Since there were no missing values so imputation was
# not needed in EDA.
# Values were consolidated based on education level as 'under-grad', ''HS-grad', 'over-grad'
# dummy variables were created for the education categories.
##############################################################################
