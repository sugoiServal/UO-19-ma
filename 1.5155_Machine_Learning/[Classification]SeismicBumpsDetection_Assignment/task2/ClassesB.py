# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 12:00:22 2019

@author: funrr
"""

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from nltk.tokenize import word_tokenize

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from skrules import SkopeRules

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris


os.chdir(r'C:\Users\funrr\Desktop\ML2')
work_dir = os.getcwd()
import ClassesA
data_dir = work_dir+'\\raw'

def ReadSeismic():
    data_dir = r'C:\Users\funrr\Desktop\ML2'  
    col = '@attribute seismic {a,b,c,d} @attribute seismoacoustic {a,b,c,d} @attribute shift {W, N} @attribute genergy real @attribute gpuls real'\
        + '@attribute gdenergy real @attribute gdpuls real @attribute ghazard {a,b,c,d} @attribute nbumps real @attribute nbumps2 real @attribute nbumps3 real'\
        + '@attribute nbumps4 real @attribute nbumps5 real @attribute nbumps6 real @attribute nbumps7 real @attribute nbumps89 real @attribute energy real'\
        + '@attribute maxenergy real @attribute class {1,0}'
    col = word_tokenize(col)    
    headers = []
    for idx, item in enumerate(col):
        if item == 'attribute':
            headers.append(col[idx + 1])
    data = pd.read_csv(data_dir +'\\seismic_bumps.csv', header = None, names=headers)
    
    data = ClassesA.Cat2Num(data)
    X = data.to_numpy()[:,0:18]
    y = data.to_numpy()[:,18]
    return X,y

def ReadIris():
    data= load_iris()
    return data['data'], data['target']

def ReadLabor():
    
    header = 'dur wage1.wage wage2.wage wage3.wage cola hours.hrs pension stby_pay shift_diff educ_allw.boolean holidays vacation lngtrm_disabil.boolean dntl_ins bereavement.boolean empl_hplan class'
    header = word_tokenize(header)  

    data = DataFrame(index=np.array(range(0,57)),columns = header)

    datapos1 = pd.read_csv(data_dir +'\\labor_pos1.csv', header = None, index_col=False, na_values = '*')
    datapos2 = pd.read_csv(data_dir +'\\labor_pos2.csv', header = None, index_col=False, na_values = '*')
    data.loc[0:18, 'dur':'pension'] = datapos1.loc[0:18, '1':'7'].values
    data.loc[0:18, 'stby_pay':'vacation'] = datapos1.loc[19:37, '1':'5'].values
    data.loc[0:18, 'lngtrm_disabil.boolean':'empl_hplan'] = datapos1.loc[38:56, '1':'4'].values
    data.loc[19:36, 'dur':'pension'] = datapos2.loc[0:17, '1':'7'].values
    data.loc[19:36, 'stby_pay':'vacation'] = datapos2.loc[18:35, '1':'5'].values
    data.loc[19:36, 'lngtrm_disabil.boolean':'empl_hplan'] = datapos2.loc[36:53, '1':'4'].values
    data.loc[0:36, 'class'] = 1

    datapos1 = pd.read_csv(data_dir +'\\labor_neg1.csv', header = None, index_col=False, na_values = '*')
    datapos2 = pd.read_csv(data_dir +'\\labor_neg2.csv', header = None, index_col=False, na_values = '*')
    data.loc[37:45, 'dur':'pension'] = datapos1.loc[0:8, '1':'7'].values
    data.loc[37:45, 'stby_pay':'vacation'] = datapos1.loc[9:17, '1':'5'].values
    data.loc[37:45, 'lngtrm_disabil.boolean':'empl_hplan'] = datapos1.loc[18:26, '1':'4'].values
    data.loc[46:56, 'dur':'pension'] = datapos2.loc[0:10, '1':'7'].values
    data.loc[46:56, 'stby_pay':'vacation'] = datapos2.loc[11:21, '1':'5'].values
    data.loc[46:56, 'lngtrm_disabil.boolean':'empl_hplan'] = datapos2.loc[22:32, '1':'4'].values
    data.loc[37:56, 'class'] = 0
 
    
    v = data.apply(lambda x: x.count(), axis=0)                 #drop column with too much missing value
    for index, row in v.iteritems():                    
        if row < 30:
            data.drop(index, axis=1, inplace = True)
            
    
        
    
    data.replace('none',0, inplace=True)
    data.replace('false',0, inplace=True)
    data.replace('true',1, inplace=True)
    data.replace('half',1, inplace=True)
    data.replace('full',2, inplace=True)
    data.replace('ba',0, inplace=True)
    data.replace('avg',1, inplace=True)
    data.replace('gnr',2, inplace=True)
    data.replace('ret_allw',1, inplace=True)
    data.replace('empl_contr',2, inplace=True)
    data.replace('tcf',1, inplace=True)
    data.replace('tc',2, inplace=True)
    data = data.astype('float64')
    #missing values??????
    
    
    mean_impute = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_t = mean_impute.fit_transform(data)
    X = data_t[:,:data_t.shape[1]-1]
    y = data_t[:,data_t.shape[1]-1:data_t.shape[1]].flatten()
    
    return X, y

    
def ReadCongress():
    header = 'Class handicapped-infants      water-project-cost-sharing      adoption-of-the-budget-resolution     physician-fee-freeze      el-salvador-aid      religious-groups-in-schools      anti-satellite-test-ban      aid-to-nicaraguan-contras     mx-missile     immigration     synfuels-corporation-cutback     education-spending     superfund-right-to-sue     crime     duty-free-exports     export-administration-act-south-africa'
    header = word_tokenize(header)  
    data = pd.read_csv(data_dir +'\\house-votes-84_data.csv', header = None, names=header)
    data.replace('y',1, inplace=True)
    data.replace('n',0, inplace=True)
    data.replace('?',0.5, inplace=True)
    data.replace('democrat',0, inplace=True)
    data.replace('republican',1, inplace=True)
    y = data.to_numpy()[:,0]
    X = data.to_numpy()[:,1:]
    
    return X, y

def KfoldAcc(X, y, multiclass = False, k = 10):                                  #then oversample pos    
    kf = KFold(n_splits=10, shuffle = True)
    accuracy = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]    #test set

        
        neigh = KNeighborsClassifier()
        neigh.fit(X_train, y_train)
        neigh_y_pred = neigh.predict(X_test)
        accuracy['neigh'].append(accuracy_score(y_test, neigh_y_pred, normalize=True, sample_weight=None))

        
        print('---------')
        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)
        tree_y_pred = tree.predict(X_test)
        accuracy['tree'].append(accuracy_score(y_test, tree_y_pred, normalize=True, sample_weight=None))

        
        print('---------')       
        naive = GaussianNB()
        naive.fit(X_train, y_train)
        naive_y_pred = naive.predict(X_test)
        accuracy['naive'].append(accuracy_score(y_test, naive_y_pred, normalize=True, sample_weight=None))

        
        print('---------')
        rule = SkopeRules()
        if multiclass is True:
            rule = OneVsRestClassifier(rule)
        rule.fit(X_train, y_train)
        rules_y_pred = rule.predict(X_test)
        accuracy['rule'].append(accuracy_score(y_test, rules_y_pred, normalize=True, sample_weight=None))


    return accuracy
        
