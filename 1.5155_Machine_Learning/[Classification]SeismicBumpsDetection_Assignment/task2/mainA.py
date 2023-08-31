# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:16:32 2019

@author: funrr
"""

import os
import numpy as np
import pandas as pd
#from nltk.tokenize import word_tokenize
#from sklearn import preprocessing
#from matplotlib import pyplot as plt
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

from pandas import DataFrame






os.chdir(r"C:\Users\funrr\Desktop\ML2")
work_dir = os.getcwd()
import ClassesA


data = ClassesA.ReadData()
ClassesA.CheckMissing(data)
data = ClassesA.Cat2Num(data)

acc = DataFrame(index = ['OverSampling', 'UnderSampling', 'BalancedSampling'], columns = pd.Index(['k-neighbor','tree','naive', 'rule'], name = 'models'))
recall = DataFrame(index = ['OverSampling', 'UnderSampling', 'BalancedSampling'], columns = pd.Index(['k-neighbor','tree','naive', 'rule'], name = 'models'))
auc = DataFrame(index = ['OverSampling', 'UnderSampling', 'BalancedSampling'], columns = pd.Index(['k-neighbor','tree','naive', 'rule'], name = 'models'))
o_accuracy, o_recall, o_auc = ClassesA.OverSampleKfold(data)
u_accuracy, u_recall, u_auc = ClassesA.UnderSampleKfold(data)
b_accuracy, b_recall, b_auc = ClassesA.BalanceSampleKfold(data)

acc.loc['OverSampling'] = list(map(lambda x: sum(o_accuracy[x])/len(o_accuracy[x]), o_accuracy))
acc.loc['UnderSampling'] = list(map(lambda x: sum(u_accuracy[x])/len(u_accuracy[x]), u_accuracy))
acc.loc['BalancedSampling'] = list(map(lambda x: sum(b_accuracy[x])/len(b_accuracy[x]), b_accuracy))


recall.loc['OverSampling'] = list(map(lambda x: sum(o_recall[x])/len(o_recall[x]), o_recall))
recall.loc['UnderSampling'] = list(map(lambda x: sum(u_recall[x])/len(u_recall[x]), u_recall))
recall.loc['BalancedSampling'] = list(map(lambda x: sum(b_recall[x])/len(b_recall[x]), b_recall))


auc.loc['OverSampling'] = list(map(lambda x: sum(o_auc[x])/len(o_auc[x]), o_auc))
auc.loc['UnderSampling'] = list(map(lambda x: sum(u_auc[x])/len(u_auc[x]), u_auc))
auc.loc['BalancedSampling'] = list(map(lambda x: sum(b_auc[x])/len(b_auc[x]), b_auc))

work_dir = os.getcwd()
plotdir = work_dir + "\\plots\\"
ax = acc.plot(kind = 'bar', rot=0)
fig = ax.get_figure()
fig.savefig(plotdir +'acc' + '.jpg', dpi = 600)

ax = recall.plot(kind = 'bar', rot=0)
fig = ax.get_figure()
fig.savefig(plotdir +'recall' + '.jpg', dpi = 600)

ax = auc.plot(kind = 'bar', rot=0)
fig = ax.get_figure()
fig.savefig(plotdir +'auc' + '.jpg', dpi = 600)




for index, item in b_accuracy.items():
    
    print(index, item)

print(ttest_ind(b_accuracy['naive'], b_accuracy['neigh'], axis=0, equal_var=False, nan_policy='propagate'))
print(ttest_ind(b_accuracy['naive'], b_accuracy['rule'], axis=0, equal_var=False, nan_policy='propagate'))
print(ttest_ind(b_accuracy['naive'], b_accuracy['tree'], axis=0, equal_var=False, nan_policy='propagate'))
print(ttest_ind(b_accuracy['neigh'], b_accuracy['rule'], axis=0, equal_var=False, nan_policy='propagate'))
print(ttest_ind(b_accuracy['neigh'], b_accuracy['tree'], axis=0, equal_var=False, nan_policy='propagate'))
print(ttest_ind(b_accuracy['rule'], b_accuracy['tree'], axis=0, equal_var=False, nan_policy='propagate'))



#X_train, X_test, y_train, y_test = train_test_split(\       #split test set and training set
#     X, y, test_size=0.25)

#---------------------------------------------

data_np = data.to_numpy()
X = data_np[:,0:18]
y = data_np[:,18]
rus_balance = RandomUnderSampler(sampling_strategy = 0.20)  #truncate neg to 5*#pos
sm_balance = SMOTE()  
kf = KFold(n_splits=10, shuffle = True)
accuracy_no = {'neigh':[],'tree':[],'naive':[],'rule':[] }    
accuracy_f_classif = {'neigh':[],'tree':[],'naive':[],'rule':[] }    
accuracy_Boruta = {'neigh':[],'tree':[],'naive':[],'rule':[] }  


##################  
X_bal, y_bal = rus_balance.fit_resample(X, y) 
X_bal, y_bal = sm_balance.fit_resample(X_bal, y_bal)





from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
X1 = feat_selector.fit_transform(X_bal, y_bal)

from sklearn.feature_selection import SelectKBest, f_classif
X2 = SelectKBest(f_classif, k=20).fit_transform(X_bal, y_bal)


#no feature sel
for train_index, test_index in kf.split(X_bal):
    X_train, X_test = X_bal[train_index], X_bal[test_index]
    y_train, y_test = y_bal[train_index], y_bal[test_index]    #test set
    
    neigh = KNeighborsClassifier()  ##
    tree = DecisionTreeClassifier()
    naive = GaussianNB()
    rule = SkopeRules(feature_names= data.columns.to_list()[0:18])
    
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    accuracy_no['neigh'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy_no['tree'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    naive.fit(X_train, y_train)
    y_pred = naive.predict(X_test)
    accuracy_no['naive'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    rule.fit(X_train, y_train)
    y_pred = rule.predict(X_test)
    accuracy_no['rule'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    
#feature sel w/ boruta
for train_index, test_index in kf.split(X1):
    X_train, X_test = X1[train_index], X1[test_index]
    y_train, y_test = y_bal[train_index], y_bal[test_index]    #test set
    
    neigh = KNeighborsClassifier()  ##
    tree = DecisionTreeClassifier()
    naive = GaussianNB()
    rule = SkopeRules()
    
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    accuracy_Boruta['neigh'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy_Boruta['tree'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    naive.fit(X_train, y_train)
    y_pred = naive.predict(X_test)
    accuracy_Boruta['naive'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    rule.fit(X_train, y_train)
    y_pred = rule.predict(X_test)
    accuracy_Boruta['rule'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
#feature sel w/ f_classif
for train_index, test_index in kf.split(X2):
    X_train, X_test = X2[train_index], X2[test_index]
    y_train, y_test = y_bal[train_index], y_bal[test_index]    #test set
    
    neigh = KNeighborsClassifier()  ##
    tree = DecisionTreeClassifier()
    naive = GaussianNB()
    rule = SkopeRules()
    
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    accuracy_f_classif['neigh'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    accuracy_f_classif['tree'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    naive.fit(X_train, y_train)
    y_pred = naive.predict(X_test)
    accuracy_f_classif['naive'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    rule.fit(X_train, y_train)
    y_pred = rule.predict(X_test)
    accuracy_f_classif['rule'].append(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
    
    

print(ttest_ind(accuracy_no['naive'], accuracy_Boruta['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(accuracy_no['naive'], accuracy_f_classif['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(accuracy_Boruta['naive'], accuracy_f_classif['naive'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')

print(ttest_ind(accuracy_no['neigh'], accuracy_Boruta['neigh'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(accuracy_no['neigh'], accuracy_f_classif['neigh'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(accuracy_Boruta['neigh'], accuracy_f_classif['neigh'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')

print(ttest_ind(accuracy_no['rule'], accuracy_Boruta['rule'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(accuracy_no['rule'], accuracy_f_classif['rule'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(accuracy_Boruta['rule'], accuracy_f_classif['rule'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')


print(ttest_ind(accuracy_no['tree'], accuracy_Boruta['tree'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(accuracy_no['tree'], accuracy_f_classif['tree'], axis=0, equal_var=False, nan_policy='propagate'))
print('_____________________________')
print(ttest_ind(accuracy_Boruta['tree'], accuracy_f_classif['tree'], axis=0, equal_var=False, nan_policy='propagate'))



        

