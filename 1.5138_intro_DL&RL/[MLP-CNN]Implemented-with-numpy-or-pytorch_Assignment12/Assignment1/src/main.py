# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:22:23 2019

@author: funrr
"""
import os
os.chdir(r"C:\Users\Boris\Desktop\dl_assignment1")
import numpy as np
import functions
import matplotlib.pyplot as plt
import time


    
result = np.zeros((3,7,21,3))
#result = np.zeros((3,7,3,3))
i = 0
loop = 0
start_time = time.time()

for sigma in (0.01,0.1,1):   
    j = 0
    for N in (2, 5, 10, 20, 50, 100, 200):
        for d in range(21):
            Ein_bar, Eout_bar, Ebias = functions.experiment(train_size = N, \
               degree= d, sigma = sigma, trial = 30)
            n_experiment = np.array([Ein_bar, Eout_bar, Ebias])
            result[i][j][d] = n_experiment
            loop = loop+1
            print("current loop: 441/" ,loop)
            print("--- %s seconds ---" % (time.time() - start_time))
        j = j+1
    i = i+1

np.save("result", result)
print("over all time")
print("--- %s seconds ---" % (time.time() - start_time))