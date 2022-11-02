# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:45:32 2019

@author: funrr
"""
from __future__ import print_function
import classes
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import normaltest, ttest_ind
from pandas import DataFrame



os.chdir(r"C:\Users\funrr\Desktop\dl assignment 2\Q3")
data_dir = os.getcwd()


train_loader, test_loader, train_all, test_all = classes.MNISTLoader()

train_label = DataFrame()
test_label = DataFrame()

train_label['training_set'] = train_all.targets
test_label['test_set'] = test_all.targets
print(train_label.describe())
print(test_label.describe())
train_label.boxplot()
plt.show()
train_label.hist()
plt.show()
test_label.boxplot()
plt.show()
test_label.hist()

d1 = np.load('MLPdrop_0_BREAD_experiment(2).npy')
d1 = np.mean(d1, axis = 0)
d1 = d1[::4]

q1 = np.load('Softdrop0BN0_experiment.npy')
q2 = np.load('Softdrop1BN0_experiment.npy')
q3 = np.load('Softdrop0BN1_experiment.npy')
q4 = np.load('MLPdrop0BN0_experiment.npy')
q5 = np.load('MLPdrop1BN0_experiment.npy')
q6 = np.load('MLPdrop0BN1_experiment.npy')
q7 = np.load('CNNdrop0BN0_experiment.npy')
q8 = np.load('CNNdrop1BN0_experiment.npy')
q9 = np.load('CNNdrop0BN1_experiment.npy')

result  = DataFrame()
result["q8"] =q8
result["q5"] =q5

normaltest(q1)
normaltest(q2)
normaltest(q3)
normaltest(q4)
normaltest(q5)   #
normaltest(q6)
normaltest(q7)
normaltest(q8)
normaltest(q9)

w1 = np.load('MLPdrop_0_DEEP_experiment(1).npy')
w2 = np.load('MLPdrop_0_DEEP_experiment(2).npy')
w3 = np.load('MLPdrop_0_DEEP_experiment(3).npy')
w4 = np.load('MLPdrop_1_DEEP_experiment(1).npy')
w5 = np.load('MLPdrop_1_DEEP_experiment(2).npy')
w6 = np.load('MLPdrop_1_DEEP_experiment(3).npy')
w7 = np.load('MLPdrop_0_BREAD_experiment(1).npy')
w8 = np.load('MLPdrop_0_BREAD_experiment(2).npy')
w9 = np.load('MLPdrop_0_BREAD_experiment(3).npy')

w2 = np.mean(w2, axis = 0)
w3 = np.mean(w3, axis = 0)
w5 = np.mean(w5, axis = 0)
w6 = np.mean(w6, axis = 0)
w8 = np.mean(w8, axis = 0)
w9 = np.mean(w9, axis = 0)
w2 = w2[::4]
w5 = w5[::4]
w8 = w8[::4]


fig1,axes= plt.subplots(1,1)
ax1 = axes
ax1.plot(w2,'--', label='MLP2_train')
ax1.plot(w3,'--', label='MLP2_test')
ax1.plot(w8, label='MLP1_train')
ax1.plot(w9, label='MLP1_test')
ax1.legend(prop={'size': 9})
ax1.set_title('MLP2 vs MLP1')
ax1.set_xlabel("1/5 epoch")
ax1.set_ylabel("losses")
fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig('ff1',dpi = 800)
fig2,axes2= plt.subplots(1,1)
ax2 = axes2
ax2.plot(w2,'--' ,label='MLP2_train')
ax2.plot(w3,'--', label='MLP2_test')
ax2.plot(w5, label='MLP3_train')
ax2.plot(w6, label='MLP3_test')
ax2.legend(prop={'size': 9})
ax2.set_title('MLP2 vs MLP3')
ax2.set_xlabel("1/5 epoch")
ax2.set_ylabel("losses")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig('ff2',dpi = 800)

f1 = np.sqrt(np.sum(np.power((w2-w3),2)))
f2 = np.sqrt(np.sum(np.power((w5-w6),2)))