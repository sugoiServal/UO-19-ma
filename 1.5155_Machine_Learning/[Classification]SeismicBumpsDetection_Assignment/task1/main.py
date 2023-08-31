# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:16:32 2019

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
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
#from sklearn.metrics import roc_curve


from sklearn.datasets import load_iris


os.chdir(r"C:\Users\Boris\Desktop\machine learning\assignment 2\ML2")
work_dir = os.getcwd()
import Classes


data = Classes.ReadData()
Classes.CheckMissing(data)
data = Classes.Cat2Num(data)


data_np = data.to_numpy()
X = data_np[:,0:18]
y = data_np[:,18]

#X_train, X_test, y_train, y_test = train_test_split(\       #split test set and training set
#     X, y, test_size=0.25)


def OverSampleKfold(k = 10):
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
        #nei scorer
        recall['neigh'].append(recall_score(y_test, neigh_y_pred, labels=None, pos_label=1))
        accuracy['neigh'].append(accuracy_score(y_test, neigh_y_pred, normalize=True, sample_weight=None))
        
        print('---------')
        tree = DecisionTreeClassifier()
        tree.fit(X_over, y_over)
        tree_y_pred = neigh.predict(X_test)
        #nei scorer
        recall['tree'].append(recall_score(y_test, tree_y_pred, labels=None, pos_label=1))
        accuracy['tree'].append(accuracy_score(y_test, tree_y_pred, normalize=True, sample_weight=None))
        
        #roc

        print('---------')       
        naive = GaussianNB()
        naive.fit(X_over, y_over)
        naive_y_pred = naive.predict(X_test)
        #nei scorer
        recall['naive'].append(recall_score(y_test, naive_y_pred, labels=None, pos_label=1))
        accuracy['naive'].append(accuracy_score(y_test, naive_y_pred, normalize=True, sample_weight=None))
        
        print('---------')
        rule = SkopeRules(feature_names= data.columns.to_list()[0:18])
        rule.fit(X_over, y_over)
        rules_y_pred = rule.predict(X_test)
        recall['rule'].append(recall_score(y_test, rules_y_pred, labels=None, pos_label=1))
        accuracy['rule'].append(accuracy_score(y_test, rules_y_pred, normalize=True, sample_weight=None))

    return accuracy, recall
        
        
def UnderSampleKfold(k = 10):
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
        #nei scorer
        recall['neigh'].append(recall_score(y_test, neigh_y_pred, labels=None, pos_label=1))
        accuracy['neigh'].append(accuracy_score(y_test, neigh_y_pred, normalize=True, sample_weight=None))
        
        print('---------')
        tree = DecisionTreeClassifier()
        tree.fit(X_under, y_under)
        tree_y_pred = neigh.predict(X_test)
        #nei scorer
        recall['tree'].append(recall_score(y_test, tree_y_pred, labels=None, pos_label=1))
        accuracy['tree'].append(accuracy_score(y_test, tree_y_pred, normalize=True, sample_weight=None))
        
        print('---------')       
        naive = GaussianNB()
        naive.fit(X_under, y_under)
        naive_y_pred = naive.predict(X_test)
        #nei scorer
        recall['naive'].append(recall_score(y_test, naive_y_pred, labels=None, pos_label=1))
        accuracy['naive'].append(accuracy_score(y_test, naive_y_pred, normalize=True, sample_weight=None))
        
        print('---------')
        rule = SkopeRules(feature_names= data.columns.to_list()[0:18])
        rule.fit(X_under, y_under)
        rules_y_pred = rule.predict(X_test)
        recall['rule'].append(recall_score(y_test, rules_y_pred, labels=None, pos_label=1))
        accuracy['rule'].append(accuracy_score(y_test, rules_y_pred, normalize=True, sample_weight=None))

    return accuracy, recall
        
        
def BalanceSampleKfold(k = 10):
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
        #nei scorer
        recall['neigh'].append(recall_score(y_test, neigh_y_pred, labels=None, pos_label=1))
        accuracy['neigh'].append(accuracy_score(y_test, neigh_y_pred, normalize=True, sample_weight=None))
        
        print('---------')
        tree = DecisionTreeClassifier()
        tree.fit(X_bal, y_bal)
        tree_y_pred = neigh.predict(X_test)
        #nei scorer
        recall['tree'].append(recall_score(y_test, tree_y_pred, labels=None, pos_label=1))
        accuracy['tree'].append(accuracy_score(y_test, tree_y_pred, normalize=True, sample_weight=None))
        
        print('---------')       
        naive = GaussianNB()
        naive.fit(X_bal, y_bal)
        naive_y_pred = naive.predict(X_test)
        #nei scorer
        recall['naive'].append(recall_score(y_test, naive_y_pred, labels=None, pos_label=1))
        accuracy['naive'].append(accuracy_score(y_test, naive_y_pred, normalize=True, sample_weight=None))
        
        print('---------')
        rule = SkopeRules(feature_names= data.columns.to_list()[0:18])
        rule.fit(X_bal, y_bal)
        rules_y_pred = rule.predict(X_test)
        recall['rule'].append(recall_score(y_test, rules_y_pred, labels=None, pos_label=1))
        accuracy['rule'].append(accuracy_score(y_test, rules_y_pred, normalize=True, sample_weight=None))

    return accuracy, recall
        
        
        
        
"""

                 #oversampler
X_res, y_res = sm.fit_resample(X, y)


neigh = KNeighborsClassifier()
neigh.fit(X[0:2580],y[0:2580])
neigh.predict(X[2580:])
y[2580:]

tree = DecisionTreeClassifier()
tree.fit(X[0:2580],y[0:2580])
tree.predict(X[2580:])
tree.predict_proba(X[2580:])

naive = GaussianNB()
naive.fit(X[0:2580],y[0:2580])
naive.predict(X[2580:])
naive.predict(X[2580:])

rule = SkopeRules(feature_names= data.columns.to_list()[0:18])
rule.fit(X[0:2580],y[0:2580])
skope_rules_scoring = rule.score_top_rules(X[2580:])



cv_results = cross_validate(naive, X, y, scoring=['accuracy','f1','precision','recall','roc_auc'], cv=10)
StratifiedKFold

"""