# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:24:08 2019

@author: funrr
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:36:54 2019

@author: funrr
"""

import os
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from skrules import SkopeRules

from sklearn.model_selection import KFold

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
#from sklearn.metrics import roc_curve

os.chdir(r"C:\Users\funrr\Desktop\ML2")
work_dir = os.getcwd()





def ReadData():
    work_dir = os.getcwd()
    col = '@attribute seismic {a,b,c,d} @attribute seismoacoustic {a,b,c,d} @attribute shift {W, N} @attribute genergy real @attribute gpuls real'\
        + '@attribute gdenergy real @attribute gdpuls real @attribute ghazard {a,b,c,d} @attribute nbumps real @attribute nbumps2 real @attribute nbumps3 real'\
        + '@attribute nbumps4 real @attribute nbumps5 real @attribute nbumps6 real @attribute nbumps7 real @attribute nbumps89 real @attribute energy real'\
        + '@attribute maxenergy real @attribute class {1,0}'
    col = word_tokenize(col)    
    headers = []
    for idx, item in enumerate(col):
        if item == 'attribute':
            headers.append(col[idx + 1])
    data = pd.read_csv(work_dir +'\\seismic_bumps.csv', header = None, names=headers)
    return data
    
    
    

def CheckMissing(data):
    #data[data.isnull().any(axis=1)]  #OR use
    missing_tag = 0
    for (columnName, columnData) in data.iteritems():    
        nullidx = data.index[data.isnull()[columnName] == True].tolist()
        if len(nullidx) != 0: 
            print('Missing Colunm Name : ', columnName)
            print('Missing idx: ')
            print(*nullidx, sep = ", ")  
            missing_tag = 1

    if missing_tag == 0:
        print('No missing value found')
        
def DataDistVis(data):
    work_dir = os.getcwd()
    plotdir = work_dir + "\\plots\\"
    os.makedirs(plotdir)
    
    for idx, col in enumerate(data.columns):
        ax = data[col].hist() 
        fig = ax.get_figure()
        fig.savefig(plotdir + str(idx) + '.' + col +'.jpg', dpi = 600)
        plt.clf()
        
def Cat2Num(data):
    cat_nums = {"seismic":     {"a": 0, "b": 1, "c": 2, "d": 3},
                "seismoacoustic": {"a": 0, "b": 1, "c": 2, "d": 3},
                "shift":     {"W": 0, "N": 1},
                "ghazard":     {"a": 0, "b": 1, "c": 2, "d": 3}}
    data.replace(cat_nums, inplace=True)
    data = data.astype({'seismic': 'int64'}) #why
    data = data.astype({'seismoacoustic': 'int64'})
    data = data.astype({'ghazard': 'int64'})
    return data

def OverSampleKfold(data, k = 10):
    data_np = data.to_numpy()
    X = data_np[:,0:18]
    y = data_np[:,18]
    sm = SMOTE()   
    kf = KFold(n_splits=k, shuffle = True)
    accuracy = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    recall = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    auc = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]    #test set
        print('num of pos:', np.sum(y_train), ', num of neg:', y_train.size-np.sum(y_train) )
        
        X_over, y_over = sm.fit_resample(X_train, y_train)   #oversampled train set
        print('oversample:', X_over.shape, y_over.shape)
        print('---------------------------------------')
        
        
        neigh = KNeighborsClassifier()
        neigh.fit(X_over, y_over)
        neigh_y_pred = neigh.predict(X_test)
        neigh_y_score = neigh.predict_proba(X_test)[:,1]
        #nei scorer
        recall['neigh'].append(recall_score(y_test, neigh_y_pred, labels=None, pos_label=1))
        accuracy['neigh'].append(accuracy_score(y_test, neigh_y_pred, normalize=True, sample_weight=None))
        auc['neigh'].append(roc_auc_score(y_test, neigh_y_score))
        
        print('---------')
        tree = DecisionTreeClassifier()
        tree.fit(X_over, y_over)
        tree_y_pred = tree.predict(X_test)
        tree_y_score = tree.predict_proba(X_test)[:,1]
        #nei scorer
        recall['tree'].append(recall_score(y_test, tree_y_pred, labels=None, pos_label=1))
        accuracy['tree'].append(accuracy_score(y_test, tree_y_pred, normalize=True, sample_weight=None))
        auc['tree'].append(roc_auc_score(y_test, tree_y_score))
        
        print('---------')       
        naive = GaussianNB()
        naive.fit(X_over, y_over)
        naive_y_pred = naive.predict(X_test)
        naive_y_score = naive.predict_proba(X_test)[:,1]
        #nei scorer
        recall['naive'].append(recall_score(y_test, naive_y_pred, labels=None, pos_label=1))
        accuracy['naive'].append(accuracy_score(y_test, naive_y_pred, normalize=True, sample_weight=None))
        auc['naive'].append(roc_auc_score(y_test, naive_y_score))
        
        print('---------')
        rule = SkopeRules(feature_names= data.columns.to_list()[0:18])
        rule.fit(X_over, y_over)
        rules_y_pred = rule.predict(X_test)
        rule_y_score = rule.rules_vote(X_test)
        recall['rule'].append(recall_score(y_test, rules_y_pred, labels=None, pos_label=1))
        accuracy['rule'].append(accuracy_score(y_test, rules_y_pred, normalize=True, sample_weight=None))
        auc['rule'].append(roc_auc_score(y_test, rule_y_score))

    return accuracy, recall, auc
        
        
def UnderSampleKfold(data, k = 10):
    data_np = data.to_numpy()
    X = data_np[:,0:18]
    y = data_np[:,18]
    rus = RandomUnderSampler()   
    kf = KFold(n_splits=10, shuffle = True)
    accuracy = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    recall = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    auc = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    
    for train_index, test_index in kf.split(X):     
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]    #test set
        print('num of pos:', np.sum(y_train), ', num of neg:', y_train.size-np.sum(y_train) )
                
        X_under, y_under = rus.fit_resample(X_train, y_train)   #oversampled train set
        print('undersample:',X_under.shape, y_under.shape)
        print('---------------------------------------')
        
        neigh = KNeighborsClassifier()
        neigh.fit(X_under, y_under)
        neigh_y_pred = neigh.predict(X_test)
        neigh_y_score = neigh.predict_proba(X_test)[:,1]
        #nei scorer
        recall['neigh'].append(recall_score(y_test, neigh_y_pred, labels=None, pos_label=1))
        accuracy['neigh'].append(accuracy_score(y_test, neigh_y_pred, normalize=True, sample_weight=None))
        auc['neigh'].append(roc_auc_score(y_test, neigh_y_score))
        
        print('---------')
        tree = DecisionTreeClassifier()
        tree.fit(X_under, y_under)
        tree_y_pred = tree.predict(X_test)
        tree_y_score = tree.predict_proba(X_test)[:,1]
        #nei scorer
        recall['tree'].append(recall_score(y_test, tree_y_pred, labels=None, pos_label=1))
        accuracy['tree'].append(accuracy_score(y_test, tree_y_pred, normalize=True, sample_weight=None))
        auc['tree'].append(roc_auc_score(y_test, tree_y_score))
        
        print('---------')       
        naive = GaussianNB()
        naive.fit(X_under, y_under)
        naive_y_pred = naive.predict(X_test)
        naive_y_score = naive.predict_proba(X_test)[:,1]
        #nei scorer
        recall['naive'].append(recall_score(y_test, naive_y_pred, labels=None, pos_label=1))
        accuracy['naive'].append(accuracy_score(y_test, naive_y_pred, normalize=True, sample_weight=None))
        auc['naive'].append(roc_auc_score(y_test, naive_y_score))
        
        print('---------')
        rule = SkopeRules(feature_names= data.columns.to_list()[0:18])
        rule.fit(X_under, y_under)
        rules_y_pred = rule.predict(X_test)
        rule_y_score = rule.rules_vote(X_test)
        recall['rule'].append(recall_score(y_test, rules_y_pred, labels=None, pos_label=1))
        accuracy['rule'].append(accuracy_score(y_test, rules_y_pred, normalize=True, sample_weight=None))
        auc['rule'].append(roc_auc_score(y_test, rule_y_score))

    return accuracy, recall, auc
        
        
def BalanceSampleKfold(data, k = 10):
    data_np = data.to_numpy()
    X = data_np[:,0:18]
    y = data_np[:,18]
    rus_balance = RandomUnderSampler(sampling_strategy = 0.20)  #truncate neg to 5*#pos
    sm_balance = SMOTE()                                        #then oversample pos    
    kf = KFold(n_splits=10, shuffle = True)
    
    accuracy = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    recall = {'neigh':[],'tree':[],'naive':[],'rule':[] }    
    auc = {'neigh':[],'tree':[],'naive':[],'rule':[] }
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]    #test set
        print('num of pos:', np.sum(y_train), ', num of neg:', y_train.size-np.sum(y_train) )
        X_bal, y_bal = rus_balance.fit_resample(X_train, y_train)  #BALANCED SAMPLE
        print('1.under:')
        print('num of pos:', np.sum(y_bal), ', num of neg:', y_bal.size-np.sum(y_bal) )
        X_bal, y_bal = sm_balance.fit_resample(X_bal, y_bal)
        print('2.over:')
        print('num of pos:', np.sum(y_bal), ', num of neg:', y_bal.size-np.sum(y_bal) )
        print('---------------------------------------')
        
        neigh = KNeighborsClassifier()
        neigh.fit(X_bal, y_bal)
        neigh_y_pred = neigh.predict(X_test)
        neigh_y_score = neigh.predict_proba(X_test)[:,1]
        #nei scorer
        recall['neigh'].append(recall_score(y_test, neigh_y_pred, labels=None, pos_label=1))
        accuracy['neigh'].append(accuracy_score(y_test, neigh_y_pred, normalize=True, sample_weight=None))
        auc['neigh'].append(roc_auc_score(y_test, neigh_y_score))
        
        print('---------')
        tree = DecisionTreeClassifier()
        tree.fit(X_bal, y_bal)
        tree_y_pred = tree.predict(X_test)
        tree_y_score = tree.predict_proba(X_test)[:,1]
        #nei scorer
        recall['tree'].append(recall_score(y_test, tree_y_pred, labels=None, pos_label=1))
        accuracy['tree'].append(accuracy_score(y_test, tree_y_pred, normalize=True, sample_weight=None))
        auc['tree'].append(roc_auc_score(y_test, tree_y_score))
        
        print('---------')       
        naive = GaussianNB()
        naive.fit(X_bal, y_bal)
        naive_y_pred = naive.predict(X_test)
        naive_y_score = naive.predict_proba(X_test)[:,1]
        #nei scorer
        recall['naive'].append(recall_score(y_test, naive_y_pred, labels=None, pos_label=1))
        accuracy['naive'].append(accuracy_score(y_test, naive_y_pred, normalize=True, sample_weight=None))
        auc['naive'].append(roc_auc_score(y_test, naive_y_score))
        
        print('---------')
        rule = SkopeRules(feature_names= data.columns.to_list()[0:18])
        rule.fit(X_bal, y_bal)
        rules_y_pred = rule.predict(X_test)
        rule_y_score = rule.rules_vote(X_test)
        recall['rule'].append(recall_score(y_test, rules_y_pred, labels=None, pos_label=1))
        accuracy['rule'].append(accuracy_score(y_test, rules_y_pred, normalize=True, sample_weight=None))
        auc['rule'].append(roc_auc_score(y_test, rule_y_score))

    return accuracy, recall, auc
        