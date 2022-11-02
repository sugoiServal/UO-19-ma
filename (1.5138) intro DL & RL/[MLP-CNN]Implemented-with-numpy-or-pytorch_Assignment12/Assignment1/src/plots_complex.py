# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:46:52 2019

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
N=(2, 5, 10, 20, 50, 100, 200)
for i in range(3):
    for j in range(7):
        ax = axes[i][j]
        if i == 2:
            ax.set_ylim([0.5,4])
        else:
            ax.set_ylim([0 ,0.8])
        ax.plot(result[i,j,:,0], label='Ein')
        ax.plot(result[i,j,:,1], label='Eout')
        ax.plot(result[i,j,:,2], label='Ebias')
        ax.legend(prop={'size': 9})
        title = "sigma="+str(sigma[i])+', N='+str(N[j])
        ax.set_title(title)
        ax.set_xlabel("degrees")
        ax.set_ylabel("MSE losses")
    

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('complex',dpi = 1000)


result = np.load("result_reg.npy")
fig,axes= plt.subplots(3,7)
fig.set_size_inches(32,12)
sigma=(0.01,0.1,1)
N=(2, 5, 10, 20, 50, 100, 200)
for i in range(3):
    for j in range(7):
        ax = axes[i][j]
        if i == 2:
            ax.set_ylim([0.025,3.5])
        else:
            ax.set_ylim([0.015,0.8])
        ax.plot(result[i,j,:,0], label='Ein')
        ax.plot(result[i,j,:,1], label='Eout')
        ax.plot(result[i,j,:,2], label='Ebias')
        ax.legend(prop={'size': 9})
        title = "sigma="+str(sigma[i])+', N='+str(N[j])
        ax.set_title(title)
        ax.set_xlabel("degrees")
        ax.set_ylabel("MSE losses")
    

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('reg_complex',dpi = 1000)
     