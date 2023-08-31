# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:53:44 2019

@author: Boris
"""

import os
os.chdir(r"C:\Users\Boris\Desktop\dl_assignment1")
import numpy as np
import matplotlib.pyplot as plt


result = np.load("result.npy")
fig,axes= plt.subplots(3,7)
fig.set_size_inches(32,12)
sigma=(0.01,0.1,1)
degree=(0,1,2,3,5,10,20)
for i in range(3):
    for j in range(7):
        ax = axes[i][j]
        if i == 2:
            ax.set_ylim([0,15])
        elif i == 1:
            ax.set_ylim([0 ,4.5])
        else:
            ax.set_ylim([0 ,3])
        ax.plot(result[i,:,degree[j],0], label='Ein')
        ax.plot(result[i,:,degree[j],1], label='Eout')
        ax.plot(result[i,:,degree[j],2], label='Ebias')
        ax.legend(prop={'size': 9})
        title = "sigma="+str(sigma[i])+', degree='+str(degree[j])
        ax.set_title(title)
        ax.set_xticklabels([2, 5, 10, 20, 50, 100, 200])
        ax.set_xlabel("#data")
        ax.set_ylabel("MSE losses")
    

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('Ndata',dpi = 1000)

result = np.load("result_reg.npy")
fig,axes= plt.subplots(3,7)
fig.set_size_inches(32,12)
sigma=(0.01,0.1,1)
degree=(0,1,2,3,5,10,20)
for i in range(3):
    for j in range(7):
        ax = axes[i][j]
        if i == 2:
            ax.set_ylim([0,15])
        elif i == 1:
            ax.set_ylim([0 ,4.5])
        else:
            ax.set_ylim([0 ,3])
        ax.plot(result[i,:,degree[j],0], label='Ein')
        ax.plot(result[i,:,degree[j],1], label='Eout')
        ax.plot(result[i,:,degree[j],2], label='Ebias')
        ax.legend(prop={'size': 9})
        title = "sigma="+str(sigma[i])+', degree='+str(degree[j])
        ax.set_title(title)
        ax.set_xticklabels([2, 5, 10, 20, 50, 100, 200])
        ax.set_xlabel("#data")
        ax.set_ylabel("MSE losses")
    

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('reg_Ndata',dpi = 1000)
