# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:57:32 2019

@author: Boris
"""


import os
os.chdir(r"C:\Users\Boris\Desktop\dl_assignment1")
import numpy as np
import matplotlib.pyplot as plt


result = np.load("result.npy")
fig,axes= plt.subplots(5,6)
fig.set_size_inches(22,12)
N=(2, 5, 10, 50, 200)
Nidx = [0,1,2,4,6]
degree=(0,1,2,5,10,20)
for i in range(5):
    for j in range(6):
        ax = axes[i][j]

        if i == 0:
            ax.set_ylim([0,10])
        else:
            ax.set_ylim([0 ,2.5])

        ax.plot(result[:,Nidx[i],degree[j],0], label='Ein')
        ax.plot(result[:,Nidx[i],degree[j],1], label='Eout')
        ax.plot(result[:,Nidx[i],degree[j],2], label='Ebias')
        ax.legend(prop={'size': 9})
        title = "N="+str(N[i])+', degree='+str(degree[j])
        ax.set_title(title)
        ax.set_xticklabels(['0.01','','0.1','','1'])
        ax.set_xlabel("sigma(noise)")
        ax.set_ylabel("MSE losses")
    

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('noise',dpi = 1000)

result = np.load("result_reg.npy")
fig,axes= plt.subplots(5,6)
fig.set_size_inches(22,12)
N=(2, 5, 10, 50, 200)
Nidx = [0,1,2,4,6]
degree=(0,1,2,5,10,20)
for i in range(5):
    for j in range(6):
        ax = axes[i][j]

        if i == 0:
            ax.set_ylim([0,10])
        else:
            ax.set_ylim([0 ,2.5])
 
        ax.plot(result[:,Nidx[i],degree[j],0], label='Ein')
        ax.plot(result[:,Nidx[i],degree[j],1], label='Eout')
        ax.plot(result[:,Nidx[i],degree[j],2], label='Ebias')
        ax.legend(prop={'size': 9})
        title = "N="+str(N[i])+', degree='+str(degree[j])
        ax.set_title(title)
        ax.set_xticklabels(['0.01','','0.1','','1'])
        ax.set_xlabel("sigma(noise)")
        ax.set_ylabel("MSE losses")
    

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('reg_noise',dpi = 1000)
