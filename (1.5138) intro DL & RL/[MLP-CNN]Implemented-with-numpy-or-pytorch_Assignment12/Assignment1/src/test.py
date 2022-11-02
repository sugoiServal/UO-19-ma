# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:36:26 2019

@author: funrr
"""
import os
os.chdir(r"C:\Users\funrr\Desktop\dl_assignment1")
import numpy as np
import functions
import matplotlib.pyplot as plt
import time
"""
result = np.zeros((7,11,3))
i = 0
loop = 0
start_time = time.time()
for reg in (0.1,0.2,0.3,0.4,0.5,0.8,1):
    for d in range(0, 21, 2):
        Ein_bar, Eout_bar, Ebias = functions.experiment(train_size = 20, \
               degree= d, sigma = 0.1, trial = 4, use_regularize= True, reg_penal = reg)
        n_experiment = np.array([Ein_bar, Eout_bar, Ebias])
        result[i][d] = n_experiment
        loop = loop+1
        print("current loop: 441/" ,loop)
        print("--- %s seconds ---" % (time.time() - start_time))
    i = i+1
np.save("tttt_reg", result)  
"""

result = np.load("tttt_reg.npy")
result = result[:7,::2,:]
fig,axes= plt.subplots(2,4)
fig.set_size_inches(30,18)
reg =( 0.1,0.2,0.3,0.4,0.5,0.8,()
for i in range(2):
    for j in range (4):
        if i+j<8:
            ax = axes[i][j]
            ax.plot(result[i+j,:,0], label='Ein')
            ax.plot(result[i+j,:,1], label='Eout')
            ax.plot(result[i+j,:,2], label='Ebias')
            ax.legend(prop={'size': 9})
            title = "sigma="+str(sigma[i])+', N='+str(N[j])
            ax.set_title(title)
            ax.set_xlabel("degrees")
            ax.set_ylabel("MSE losses")
    

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('ff',dpi = 1000)

start_time = time.time()
example = enumerate(tld)
step, tbat, lab = next(example)
print("--- %s seconds ---" % (time.time() - start_time))