#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 12:24:29 2018

@author: sulbha
"""

#############################################################################
# 1- import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
#############################################################################

#############################################################################
# 2,3 - load data from file to pandas dataframe and assign column names
file_name = "Indian Liver Patient Dataset (ILPD).csv"
column_name = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt','Sgot', 'TP', 'ALG', 'A/G', 'Selector']
data = pd.read_csv( file_name,header = None, names=column_name)
#############################################################################

#############################################################################
# Get the understanding of data
print (data.head())
print (data.dtypes)
print (data.info())
print (data.describe())
print (data.shape)
#############################################################################

#############################################################################
# 4 - Data Exploration by histogram of all variables

## plot histogram for Age
data.loc[:,'Age'].hist()
plt.xlabel('Age')
plt.title('Age Distribution')
plt.show()

# plot histogram for Gender
plt.hist(data.loc[:,'Gender'])
plt.xlabel('Gender')
plt.title('Gender Distribution')
plt.show()

# plot histogram for TB
# 'TB' has float values. As histogram cannot be developed on such data so converting to integer.
plt.hist(data.loc[:,'TB'].astype(int))
plt.xlabel('TB')
plt.title('Total Bilirubin Distribution')
plt.show()

# plot histogram for DB
# 'DB' has float values. As histogram cannot be developed on such data so converting to integer.
plt.hist(data.loc[:,'DB'].astype(int))
plt.xlabel('DB')
plt.title('Direct Bilirubin Distribution')
plt.show()

# plot histogram for Alkphos
plt.hist(data.loc[:,'Alkphos'])
plt.xlabel('Alkphos')
plt.title('Alkaline Phosphotase  Distribution')
plt.show()

# plot histogram for Sgpt
plt.hist(data.loc[:,'Sgpt'])
plt.xlabel('Sgpt')
plt.title('Alamine Aminotransferase Distribution')
plt.show()

# plot histogram for Sgot
plt.hist(data.loc[:,'Sgot'])
plt.xlabel('Sgot')
plt.title('Aspartate Aminotransferase Distribution')
plt.show()

# plot histogram for TP
# As 'TP' has float values so converting to integer for visualization 
plt.hist(data.loc[:,'TP'].astype(int))
plt.xlabel('TP')
plt.title('Total Protiens Distribution')
plt.show()

# plot histogram for ALG
# as 'ALG' has float values so converting to integer for visualization
plt.hist(data.loc[:,'ALG'].astype(int))
plt.xlabel('ALG')
plt.title('Albumin Distribution')
plt.show()

# plot histogram for A/G
## There are null values so drop those rows, also convert  'A/G' to integer
plt.hist(data.loc[:,'A/G'].dropna().astype(int))
plt.xlabel('A/G')
plt.title('Ratio Albumin and Globulin Ratio Distribution')
plt.show()

# plot histogram for Selector
plt.hist(data.loc[:,'Selector'])
plt.xlabel('Selector')
plt.title('Selector Distribution')
plt.show()
#############################################################################

#############################################################################
# 5 - create scatter plot of all the variables
scatter_matrix( data,figsize=(10,10), s= 200)
plt.title('Scatter of all variables')
plt.show()
#############################################################################

#############################################################################
# 6 - compute standard deviation of all the variables
print("Standard deviation of all columns: \n{} ".format(np.std(data)))
#############################################################################

#############################################################################
# 7 - median imputation of the missing values
print("Sum of null values in columns-Before: \n{}".format(data.isnull().sum()))

## Column A/G has null values
nullAG = np.isnan(data.loc[:,'A/G'])
data.loc[nullAG, 'A/G'] = np.nanmedian(data.loc[:,'A/G'])

# verfify with getting sum of all nulls
print("Sum of null values in columns-After: \n{}".format(data.isnull().sum()))
#############################################################################

#############################################################################
# 8 outlier replacement for 'TB'
print ("Unique TB values", data.loc[:,'TB'].unique())

data.loc[:,'TB'].hist()
plt.xlabel('TB')
plt.title('Total Bilirubin Distribution-BEFORE outlier replacement')
plt.show()

## as values > 50 are outliers so lets replace them with mean
meanTB = np.mean(data.loc[:,'TB'])
data.loc[data.loc[:,'TB']>50,'TB'] = meanTB
data.loc[:,'TB'].hist()
plt.xlabel('TB')
plt.title('Total Bilirubin Distribution-AFTER outlier replacement')
plt.show()
#############################################################################

