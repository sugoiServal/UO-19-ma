# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:07:20 2019

@author: funrr
"""
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt

os.chdir(r'C:\Users\funrr\Desktop\ML2')
work_dir = os.getcwd()
import ClassesB
from scipy.stats import ttest_ind
from scipy.stats import friedmanchisquare


X, y = ClassesB.ReadSeismic()
acc1 = ClassesB.KfoldAcc(X,y)

X, y = ClassesB.ReadIris()   ##3-class
acc2 = ClassesB.KfoldAcc(X,y,multiclass=True)

X, y = ClassesB.ReadLabor()
acc3 = ClassesB.KfoldAcc(X,y)

X, y = ClassesB.ReadCongress()
acc4 = ClassesB.KfoldAcc(X,y)

a = np.zeros([4,4])
classfier = ['naive','neigh','rule', 'tree']

for index, classifier in enumerate(classfier):
    a[0 ,index] = np.mean(acc1[classifier])
    a[1 ,index] = np.mean(acc2[classifier])
    a[2 ,index] = np.mean(acc3[classifier])
    a[3 ,index] = np.mean(acc4[classifier])
 
from scipy.stats import friedmanchisquare



statistic, p_value = friedmanchisquare(a[:,0],a[:,1],a[:,2],a[:,3])



for index, classifier in enumerate(classfier):
    p_value[index, 0] = friedmanchisquare(acc1[classifier], acc2[classifier])[1]
    p_value[index, 1] = friedmanchisquare(acc1[classifier], acc3[classifier])[1]
    p_value[index, 2] = friedmanchisquare(acc1[classifier], acc4[classifier])[1]
    p_value[index, 3] = friedmanchisquare(acc2[classifier], acc3[classifier])[1]
    p_value[index, 4] = friedmanchisquare(acc2[classifier], acc4[classifier])[1]
    p_value[index, 5] = friedmanchisquare(acc3[classifier], acc4[classifier])[1]



"""
    p_value[index, 0] = ttest_ind(acc1[classifier], acc2[classifier], axis=0, equal_var=False, nan_policy='propagate')[1]
    p_value[index, 1] = ttest_ind(acc1[classifier], acc3[classifier], axis=0, equal_var=False, nan_policy='propagate')[1]
    p_value[index, 2] = ttest_ind(acc1[classifier], acc4[classifier], axis=0, equal_var=False, nan_policy='propagate')[1]
    p_value[index, 3] = ttest_ind(acc2[classifier], acc3[classifier], axis=0, equal_var=False, nan_policy='propagate')[1]
    p_value[index, 4] = ttest_ind(acc2[classifier], acc4[classifier], axis=0, equal_var=False, nan_policy='propagate')[1]
    p_value[index, 5] = ttest_ind(acc3[classifier], acc4[classifier], axis=0, equal_var=False, nan_policy='propagate')[1]



print('_____________________________')
print(ttest_ind(acc1['naive'], acc3['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc1['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc2['naive'], acc3['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc2['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc3['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')


print(ttest_ind(acc1['naive'], acc2['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc1['naive'], acc3['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc1['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc2['naive'], acc3['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc2['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc3['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')


print(ttest_ind(acc1['naive'], acc2['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc1['naive'], acc3['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc1['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc2['naive'], acc3['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc2['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc3['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')



print(ttest_ind(acc1['naive'], acc2['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc1['naive'], acc3['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc1['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc2['naive'], acc3['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc2['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(acc3['naive'], acc4['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
"""