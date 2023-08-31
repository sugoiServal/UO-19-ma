# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:33:33 2019

@author: funrr
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:34:06 2019

@author: funrr
"""

import os
os.chdir(r"C:\Users\funrr\Desktop\dl_assignment1")
import numpy as np
import functions
import matplotlib.pyplot as plt
import time


lr_history = np.zeros([1,6])
idx = 0
learn_rate = [0.5, 0.3, 0.25, 0.2, 0.15, 0.1]
for lr in learn_rate:
    for i in range(10):
        good_history = []
        theta, Ein, Eout = functions.fitData(6,200,0.1, learn_rate\
                                             =lr)
        good_history.append(Ein)
    temp = np.array(good_history)
    mean = np.mean(temp)
    lr_history[idx] = mean
    idx = idx+1
lr = learn_rate[lr_history.argmax()]
