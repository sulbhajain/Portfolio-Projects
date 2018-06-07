#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:42:18 2018

@author: sulbha
"""

####################################################################################
#  import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
####################################################################################


####################################################################################
# load the dataset and column names

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', \
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', \
        'hours-per-week', 'native-country', 'salary-class']

cols = [c.replace(' ', '-') for c in cols]

data = pd.read_csv(url, header = None)

data.columns = cols
####################################################################################


####################################################################################
# get the glimpse the data

print(data.head())
print(data.describe())
print(data.dtypes)
print(data.info())
####################################################################################

####################################################################################

# 1. Accout for missing and outlier values:

####################################################################################

# 1.a - handle missing values in COUNTRY column: 
###############################################

print("Look for null values count in the dataset:")
print (data.isnull().sum())

# no null values as per above, let's see unique values for the country
print("Look for unique values count for COUNTRY in the dataset:")
print(data['native-country'].value_counts())

# as there are some unknown values so replace ' ?' with nulls
data = data.replace(to_replace=' ?' , value = float("NaN"))

# check count again
print("Unique values count for COUNTRY after replacing ? with nulls:")
print (data['native-country'].value_counts())

print("Null Count in the dataset:")
print (data.isnull().sum())

# now we see null values, lets drop null values
data.dropna(axis = 0, inplace = True )
print("Count after removing nulls:")
print (data.isnull().sum())
####################################################################################

####################################################################################

# 1.b - handle outliers in 'hours per week' and 'Age'
####################################################

plt.xlabel('Hours per week')
plt.title('Distrbution of Hours per week')
data['hours-per-week'].plot(kind = 'hist', bins = 30, color = 'r')
plt.show()

# as hours per week less than 10 nd greater than 80 seems outliers so removing it 
data = data[(data['hours-per-week']>10) & (data['hours-per-week']<80)]

plt.xlabel('Hours per week')
plt.title('Distrbution of Hours per week after removing outliers')
data['hours-per-week'].plot(kind = 'hist', bins = 30, color = 'r')
plt.show()

### age
plt.xlabel('Age')
plt.title('Distrbution of Age')
data['age'].plot(kind = 'hist', bins = 30, color = 'y')
plt.show()

# as age greater than 80 seems outliers so removing it 
data = data[data['age']<80]

plt.xlabel('Age')
plt.title('Distrbution of Age after removing outlier')
data['age'].plot(kind = 'hist', bins = 30, color = 'y')
plt.show()
####################################################################################


####################################################################################

#2 - Normalize numeric values
####################################################################################

# lets z-normalize 'capital-gain' and 'capital-loss'
################################################

plt.xlabel('Capital-gain')
plt.title('Distrbution Capital gain')
data['capital-gain'].plot( kind = 'hist', bins = 50)
plt.show()

# as there are gains > 20K seems outlier so remove that
data = data.loc[data['capital-gain']<20000,: ]

plt.xlabel('Capital gain')
plt.title('Distrbution of Capital gain AFTER removing outlier')
data['capital-gain'].plot(kind ='hist'  ,bins = 50)
plt.show()

# get the capital gain series 
gain_series = data['capital-gain'].values.reshape(-1,1)

# normalize using numpy 
z1 = (gain_series - np.mean(gain_series))/np.std(gain_series)
z_df1 = pd.DataFrame(z1)
print("Numpy Normalized Gain Series description: \n", z_df1.describe())
#plt.xlabel('Normalized Capital gain')
#plt.title('Distrbution of Normalized capital gain')
#z_df1.plot(kind ='hist')
#plt.hist(z_df1, bins =50)
#plt.show()

# notrmilize using scikit package
std_scaler = StandardScaler()
std_scaler.fit(gain_series)
z = std_scaler.transform(gain_series)
z_df = pd.DataFrame(z)

print("Scikit Normalized Gain Series description: \n",z_df.describe())
#plt.xlabel('Capital Gain')
#plt.title('Distrbution of Normalized Capital Gain from scikit')
#z_df.plot(kind = 'hist')
#plt.hist(z_df, bins = 50)
#plt.show()

#######################################################
## added normalized capital-gain column to the data set
#######################################################

data['norm-capital-gain'] = z_df
print("After adding normalized caital gain column")
print (data.head())

## notmalize capital loss
plt.xlabel('Capital Loss')
plt.title('Distrbution of Capital loss')
data['capital-loss'].plot(kind = 'hist',bins = 50)
plt.show()

# as there are gains > 2K seems outlier so remove that
data = data.loc[data['capital-loss']<2000,: ]
plt.xlabel('Capital Loss')
plt.title('Distrbution of Capital loss AFTER remving outlier')
data['capital-loss'].plot(kind = 'hist' ,bins = 50)
plt.show()

# get the capital loss series 
loss_series = data['capital-loss'].values.reshape(-1,1)

# normalize using numpy 
z2 = (loss_series - np.mean(loss_series))/np.std(loss_series)
z2_df = pd.DataFrame(z1)

print("Numpy Normalization of loss series: ", z2_df.describe())

#plt.xlabel('Capital loss')
#plt.title('Distrbution of Normalized Capital loss')
#z2_df.plot(kind = 'hist')
#plt.show()

# notrmilize using scikit package
std_scaler = StandardScaler()
std_scaler.fit(loss_series)
z3 = std_scaler.transform(loss_series)
z3_df = pd.DataFrame(z)
print("Scikit Normalization of loss series: ",z3_df.describe())
#plt.xlabel('Capital loss')
#plt.title('Distrbution of Normalized capital loss')
#z3_df.plot(kind='hist')
#plt.show()

#######################################################
## added normalized capital-loss column to the data set
#######################################################

data['norm-capital-loss'] = z3_df
print("After adding normalized caital loss column")
print (data.head())

plt.title('Age Vs Normalized capital gain')
plt.scatter(data.loc[:,'age'], data.loc[:, 'norm-capital-gain'])
plt.show()

plt.title('Age Vs Normalized capital loss')
plt.scatter(data.loc[:,'age'], data.loc[:, 'norm-capital-loss'])
plt.show()
####################################################################################


####################################################################################
#3 - Bin numeric variables
####################################################################################

# lets bin age using equal width algorithm and numpy
# I = Intermediate, E = Experienced, S = Senior

numberofbins  = 3

age_series = np.array(data['age'])

BinWidth = (max(age_series)- min(age_series))/numberofbins

MinBin1 = float('-inf')
MaxBin1 = min(age_series) + 1 * BinWidth
MaxBin2 = min(age_series) + 2 * BinWidth
MaxBin3 = float('inf')

print("\n########\n\n Bin 1 is from ", MinBin1, " to ", MaxBin1)
print(" Bin 2 is greater than ", MaxBin1, " up to ", MaxBin2)
print(" Bin 2 is greater than ", MaxBin2, " up to ", MaxBin3)

Binned_EqW = np.array([" "]*len(age_series)) # Empty starting point for equal-width-binned array
Binned_EqW[(MinBin1 < age_series) & (age_series <= MaxBin1)] = "I" # intermediate
Binned_EqW[(MaxBin1 < age_series) & (age_series <= MaxBin2)] = "E" # Experienced
Binned_EqW[(MaxBin2 < age_series) & (age_series  < MaxBin3)] = "S" # Senior

print(" AGE binned into 3 equal-width bins: ")
print(Binned_EqW)

###########################################
# added binned age category to the dataset
###########################################
data['age-bin-category'] = Binned_EqW
print(data.head())

print("Count of binned AGE Class Category")
print(data['age-bin-category'].value_counts())
####################################################################################


####################################################################################
#4 - consolidate categorical data for 'workclass' as new categores - private-gov-No Income
####################################################################################
print("Count of Work Class Category")
print(data['workclass'].value_counts())
data.loc[data.loc[:, 'workclass']==' Self-emp-not-inc', 'workclass'] = ' Without-pay'
data.loc[data.loc[:, 'workclass']==' Local-gov', 'workclass'] = ' Gov'
data.loc[data.loc[:, 'workclass']==' State-gov', 'workclass'] = ' Gov'
data.loc[data.loc[:, 'workclass']==' Federal-gov', 'workclass'] = ' Gov'
data.loc[data.loc[:, 'workclass']==' Self-emp-inc', 'workclass'] = ' Private'
print("Cout after consolidating")
print(data['workclass'].value_counts())

## create dummpy variables for categoricla data
data['gov-job'] = (data.loc[:,'workclass'] == ' Gov').astype(int)
data['private-job'] = (data.loc[:,'workclass'] == ' Private').astype(int)
data['no-job'] = (data.loc[:,'workclass'] == ' Without-pay').astype(int)

print("Appended hot encoded work categories to the data")
print(data.head())
####################################################################################
 

####################################################################################
#5 - Remove obsolete cilumns
####################################################################################

# remove workforce column as it is not longer needed
fields_drop = ['capital-loss', 'capital-gain', 'workclass']
data.drop(fields_drop, axis = 1, inplace=True)
print("Data after removing Obsolete column: capital-gain, capital-loss, workclass ")
print(data.head())
####################################################################################


####################################################################################
## write the dataset to a file
data.to_csv("Sulbha-M02-Dataset.csv", encoding= 'utf-8', index=False)
import os
print("List of files stored in the directory: ", os.listdir())
####################################################################################
