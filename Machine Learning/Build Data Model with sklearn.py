#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:39:14 2018

@author: sulbha
"""

# Problem Statement - Analyze the experiment data related to patient condition and predict 
#                     whether there are chances of heart disease where 0 - no disease distinguishes absense
#                     from 1-4: presence 

##############################################################################
# 1: Have all the required libraries needed for the analysis
##############################################################################
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import confusion_matrix, classification_report


##############################################################################
# 2: Define functions to normalize and split train-test data. 
##############################################################################

# use Z-scale for the normalization
def normalize(X):
    N,m  = X.shape
    Y = np.zeros([N,m])
    for i in range(m):
        # here iloc is used to address the columns as these are dataframe index
        Y[:,i] = np.round(X.iloc[:,i]- np.mean(X.iloc[:,i])/np.std(X.iloc[:,i]))
        print ("i is" , i)
    
    return Y


def split_dataset(data, r): # split a dataset in matrix format, using a given ratio for the testing set
	N = len(data)	
	X = []
	Y = []
	
	if r >= 1: 
		print ("Parameter r needs to be smaller than 1!")
		return
	elif r <= 0:
		print ("Parameter r needs to be larger than 0!")
		return

	n = int(round(N*r)) # number of elements in testing sample
	nt = N - n # number of elements in training sample
	ind = -np.ones(n,int) # indexes for testing sample
	R = np.random.randint(N) # some random index from the whole dataset
	
	for i in range(n):
		while R in ind: R = np.random.randint(N) # ensure that the random index hasn't been used before
		ind[i] = R

	ind_ = list(set(range(N)).difference(ind)) # remaining indexes	
	X = data[ind_,:-1] # training features
	XX = data[ind,:-1] # testing features
	Y = data[ind_,-1] # training targets
	YY = data[ind,-1] # testing targets
	return X, XX, Y, YY

##############################################################################
# 1: import heath data for the purpose
##############################################################################

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'disease-presence' ]
data = pd.read_csv(url, header = None, delimiter = " ", names = names)
print (data.head())

# Perform data cleaning to remove null values, outliers etc to prepare dataset for the modelling 
# As per data source reference, -9.0 represents null value in the dataset so replacing it with 0
data = data.replace(to_replace=-9.0 , value = 0)
data = data.replace(to_replace='NA' , value = 0)
data.dropna(axis = 0,inplace=True)
print (data.head())

##############################################################################
# 2: Call the normalize funtions to normalize all columns of the dataset
#    Also distributed the dataset as train-test with 80:20 split.
##############################################################################

data_normalized = normalize(data)
r = 0.2
X, XX, Y, YY = split_dataset(data_normalized,r) 

##############################################################################
# Run all the classification models to see performance and results.
# for now used 'Classification report' and 'Confusion matrix' to validate the trained model performance on test data.
##############################################################################

""" CLASSIFICATION MODELS """
# Logistic regression classifier
print ('\n\n\nLogistic regression classifier\n')
C_parameter = 50. / len(X) # parameter for regularization of the model
class_parameter = 'ovr' # parameter for dealing with multiple classes
penalty_parameter = 'l1' # parameter for the optimizer (solver) in the function
solver_parameter = 'saga' # optimization system used
tolerance_parameter = 0.1 # termination parameter
#####################

#Training the Model
clf = LogisticRegression(C=C_parameter, multi_class=class_parameter, penalty=penalty_parameter, solver=solver_parameter, tol=tolerance_parameter)
clf.fit(X, Y) 
print ('coefficients:')
print (clf.coef_) # each row of this matrix corresponds to each one of the classes of the dataset
print ('intercept:')
print (clf.intercept_) # each element of this vector corresponds to each one of the classes of the dataset

# Apply the Model
print ('predictions for test set:')
print (clf.predict(XX))
print ('actual class values:')
print (YY)
print (confusion_matrix(YY, clf.predict(XX) ))
print (classification_report(YY, clf.predict(XX)))
#####################

# Naive Bayes classifier
print ('\n\nNaive Bayes classifier\n')
nbc = GaussianNB() # default parameters are fine
nbc.fit(X, Y)
print ("predictions for test set:")
print (nbc.predict(XX))
print ('actual class values:')
print (YY)

print (confusion_matrix(YY, clf.predict(XX) ))
print (classification_report(YY, nbc.predict(XX)))
####################

# k Nearest Neighbors classifier
print ('\n\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
knn.fit(X, Y)
print ("predictions for test set:")
print (knn.predict(XX))
print ('actual class values:')
print (YY)

print (confusion_matrix(YY, clf.predict(XX) ))
print (classification_report(YY, knn.predict(XX)))
###################

# Support vector machine classifier
t = 0.001 # tolerance parameter
kp = 'rbf' # kernel parameter
print ('\n\nSupport Vector Machine classifier\n')
clf = SVC(kernel=kp, tol=t)
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)

print (confusion_matrix(YY, clf.predict(XX) ))
print (classification_report(YY, clf.predict(XX)))
####################

# Decision Tree classifier
print ('\n\nDecision Tree classifier\n')
clf = DecisionTreeClassifier() # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)

print (confusion_matrix(YY, clf.predict(XX) ))
print (classification_report(YY, clf.predict(XX)))
####################

# Random Forest classifier
estimators = 9 # number of trees parameter
mss = 3 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)

print (confusion_matrix(YY, clf.predict(XX) ))
print (classification_report(YY, clf.predict(XX)))
####################

##############################################################################
# 3: Choice of classifier: On repeated iteration of modelling, Random Forest Classifier performs best consistently 
##############################################################################

### COMMENTS:
# after running all the classsification model it is noted that 'Random Forest classfier'
# with parameters estimatores = 9 and mss =3  
# has the highest precision (83%) and recall (80%) so that is the best model to predeict 
# presence of heart disease. However Logistic regression also works good in most cases

print("****** Random Forest is selected ****** ")
estimators = 9 # number of trees parameter
mss = 3 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
clf.fit(X, Y)
print ("predictions for test set:")
print (clf.predict(XX))
print ('actual class values:')
print (YY)

print (confusion_matrix(YY, clf.predict(XX) ))
print (classification_report(YY, clf.predict(XX)))
