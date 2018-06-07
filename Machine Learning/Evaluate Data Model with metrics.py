#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:20:06 2018

@author: sulbha
"""
# problem statment - Evaluate accuracy measures for a logistic regression Model.

##############################################################################
# 1: Have all the required libraries needed for the analysis
##############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from numpy import hstack
##############################################################################
# 1: Define functions to normalize and split train-test data. 
##############################################################################

# use Z-scale for the normalization
def normalize(X):
    N,m  = X.shape
    Y = np.zeros([N,m])
    for i in range(m):
        # here iloc is used to address the columns as these are dataframe index
        Y[:,i] =  np.round(X.iloc[:,i]- np.mean(X.iloc[:,i])/np.std(X.iloc[:,i]))
    
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

# convert 2,3 and 4 values to '1' distinguising as presence of disease from absence. 
data.loc[(data.loc[:,'disease-presence']==2) | (data.loc[:,'disease-presence']==3) | (data.loc[:,'disease-presence']==4) , 'disease-presence'] = 1

##############################################################################
# 1: Call the normalize funtions to normalize all columns of the dataset
#    Also distributed the dataset as train-test with 80:20 split.
##############################################################################


data_normalized = normalize(data.drop(['disease-presence'], axis  =1))  
y_normalized  = np.array(pd.DataFrame(data=data['disease-presence'] ,index = data.index,columns = ['disease-presence']))
data_normalized = hstack([data_normalized, y_normalized])

r = 0.2
X, XX, Y, YY = split_dataset(data_normalized,r) 


##############################################################################
# Run logistic classification models to see performance and results.
# ##############################################################################

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
predict =  (clf.predict(XX))
print ('predictions for test set:')
print (predict)
print ('actual class values:')
print (YY)

##############################################################################
# #2: run accuracy measurs for the predicted values
# ##############################################################################

# 2a: eval confusion matrix and evaluation report
print (confusion_matrix(YY, predict ))
print (classification_report(YY, predict))

cm = confusion_matrix(YY, predict)
tn, fp, fn, tp = cm.ravel()

print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
AR = accuracy_score(YY, predict)
print ("\nAccuracy rate:", AR)
ER = 1.0 - AR
print ("\nError rate:", ER)


P = precision_score(YY, predict)
print ("\nPrecision:", np.round(P, 2))
R = recall_score(YY, predict)
print ("\nRecall:", np.round(R, 2))
F1 = f1_score(YY, predict)
print ("\nF1 score:", np.round(F1, 2))
####################

# ROC analysis
LW = 1.5 # line width for plots
LL = "lower right" # legend location
LC = 'darkgreen' # Line Color

#2b-c: perform ROC analysis and probablity thresholds
fpr, tpr, th = roc_curve(YY, predict) # False Positive Rate, True Posisive Rate, probability thresholds
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))

print ("\nProbability thresholds:", clf.predict_proba(XX))
#####################

plt.figure()
plt.title('Receiver Operating Characteristic curve example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FALSE Positive Rate')
plt.ylabel('TRUE Positive Rate')
plt.plot(fpr, tpr, color=LC,lw=LW, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='navy', lw=LW, linestyle='--') # reference line for random classifier
plt.legend(loc=LL)
plt.show()
####################

print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(YY, predict), 2), "\n")

######################


