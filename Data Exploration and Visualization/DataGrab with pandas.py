#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 19:44:01 2018

@author: sulbha
"""
##############################################################################
### 1 - Import statements for all libraries needed in the code
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
##############################################################################


##############################################################################
### 2 - url of the data. As there are no header names in the data so it is not needed
## import Hepatitis data from ics URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
data = pd.read_csv(url, header = None)
##############################################################################

##############################################################################
## 3 - print first 5 rows
data.head()
##############################################################################

##############################################################################
## 4 assign featue names to the data
data_col = ['CLASS', 'AGE', 'SEX','STEROID','ANTIVIRALS', 'FATIGUE',
           'MALAISE', 'ANOREXIA', 'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE',
           'SPIDERS','ASCITES','VARICES','BILIRUBIN', 'ALK PHOSPHATE', 'SGOT',
           'ALBUMIN', 'PROTIME','HISTOLOGY']
data.columns = data_col

## have a feel of the data 
data.head()
data.info()
data.describe()

## get list of columns though not needed in this case as we assigned names through data_col 
## but good to have for regulat projects.
cols = data.columns.tolist()[:-1]
cols
##############################################################################

##############################################################################
### 5 - create histogram for numerical data

## function to create the plot. other features could be defined here for the subplots
def create_hist(df):
    for col in data_col:
        if df[col].dtype in [np.int32, np.int64, np.float]:
           df.hist(column = [col], bins = 30, figsize=(6,6))

## call the function to plot numerical data
create_hist(data)


### additional: plot by 'CLASS' column and also show countplot
def create_hist_byclass(df):
    for col in data_col:
        if df[col].dtype in [np.int32, np.int64, np.float] and df[col].name != 'CLASS':
            df.hist(column = [col], bins = 30, figsize=(6,6), by ='CLASS')
            sns.countplot(x=col, hue = 'CLASS', data = df)
## call the function to show the graph
create_hist_byclass(data)
##############################################################################

##############################################################################
## 6 plot 'bar' for categorical data. since datatype is object so plotted bargraph 
## after convertaing data to int 
def create_hist_category(df):
    df_int= pd.DataFrame()
    
    for col in data_col:
        if df[col].dtype not in [np.int32, np.int64, np.float] :
            df.plot(kind = 'bar',x = [col], figsize=(6,6), y = 'CLASS')
           
            col_name = 'int_'+ col
            df_int.loc[:,col_name]  = df.loc[:,col].replace(to_replace="?", value = float("NaN"))
            df_int.loc[:,col_name]  = pd.to_numeric( df_int.loc[:,col_name])
    
    # draw hsiogram after converting categorical data to integer
    df_int.hist( bins = 30, figsize=(10,10))

# call function to plot categorical data
create_hist_category(data)
##############################################################################

##############################################################################
## 7 create scatter plot of all numerical data, also color by 'CLASS' and specify other attributes
scatter_matrix(data ,s= 100, figsize = (10,10), c= data.loc[:,'CLASS'])
##############################################################################

##############################################################################
## 8 find missing values and count of it for 'STERIOD' and 'LIVER BIG'
data['STEROID'].value_counts()
data['LIVER BIG'].value_counts()
## count of number of unknown/missing('?') values across the entire dataset
print ("Count of ? values in all columns: \n")
data.isin(['?']).sum()
##############################################################################

##############################################################################
## 9 - identify potential outliers in numerical columns
## first check data distribution of numerical columns
data.hist()
### 'AGE' column is continuous variable so lets identify the outliers
## calculate mean and standard deviation
mean = np.mean(data['AGE'])
std = np.std(data['AGE'])

limithigh = mean + 2*std
limitlow = mean - 2*std

print ("Outliers for Age: \n")
data.loc[(data['AGE']>limithigh) | (data['AGE']<limitlow), 'AGE']
##############################################################################

##############################################################################
## SUMMARY -
## 1. Chances of dying with Hepetitis disease is more for older ages. 
## 2. Females donot die with Hepatitus B
## 3. Steroid 2 has low death rates.
## 4. There are categorical data columns e.g. 'ALK PHOSPHATE', 'ALUBIUM' etc. which 
##    should be converted to numerical data before creating the model.
## 5. All missing values need to be handled before creating the model.
## 6. Classfication model can be build to predict CLASS- LIVE, DIE based on the features
## 7. Model improvement based on binary tree method. Evaluate and predict. 
##############################################################################