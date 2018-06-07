#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:03:29 2018

@author: sulbha
"""

import numpy as np

arr_age = np.array([4, 5, " " , 6, 4, 4, "4", -1, 6, 5, 4, 4, 5, 45, 5, 5, 5, 6, 5,
                   6, 4, 99, 4, 6, "?", 6, 6, "NA", 4, 5, 4, 6, 6, 6, 6, 4, 5, 4, 
                   6, 4, 6," " , 4, 5, 5, 5])


# function to remove outlies from the array and return good array
def remove_outlier(arr1):
    
    # calculate the acceptable range    
    high_limit = np.mean(arr1) + 2 *np.std(arr1)
    low_limit = np.mean(arr1) - 2*np.std(arr1)
    good_arr = (arr1 <=high_limit )& (arr1 >= low_limit)
    
    return (arr1[good_arr])


# function to remove non-numeric values from the array
def clean_array(arr1):
    
    good_arr = [element.isdigit() for element in arr1]
    arr1 = arr1[good_arr].astype(int)
    
    return (arr1)

# function to replace outlier values with median of the array
def replace_outlier(arr1):
    high_lim = np.mean(arr1) + 2 *np.std(arr1)
    low_lim = np.mean(arr1) - 2 *np.std(arr1)
    
    bad_arr = (arr1>high_lim ) | (arr1<low_lim)
    good_arr = ~bad_arr
    
    arr1[bad_arr] = np.median(arr1[good_arr])
    
    return(arr1)

# function to default missing/NA values. 
def fill_missing(arr1):
   # find spaces in the array and impute with "0"
    missing_idx= [(x==" ") for x in arr1]
    arr1[missing_idx] = "0"

    # find "NA" in the array and impute with "0"
    missing_idx= [(x=="NA") for x in arr1]
    arr1[missing_idx] = "0"
    
    return (arr1)


# fill missing and "NA" elements with "0"
print ("original array", arr_age)
arr_age = fill_missing(arr_age)
print ("Imputed Array", arr_age)

# clean the array and keep only numeric entries
arr_age = clean_array(arr_age) 
print ("cleaned array", arr_age)
 
# remove outliers
good_array = remove_outlier(arr_age)
print ("Outliers removed", good_array)
 
# replace outliers
replace_arr = replace_outlier(arr_age)
print ("After outliers are replaced", replace_arr)
 