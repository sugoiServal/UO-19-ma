# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:47:46 2019

@author: Boris
"""

import numpy as np
import os
os.chdir(r"C:\Users\funrr\Desktop\dl assignment 2\Q1")

import classes
from classes import network
import matplotlib.pyplot as plt

k = 50
N = 100
learning_rate = 0.01
x = np.random.randn(k, N)
net = network(k)
Loss = []
loss = net.ForwardLoss(x)
diffA, diffB, loss = net.BackwardGradient(x)

#diffA, diffB, loss = net.BackwardGradient(x)

for i in range(15000):
    diffA, diffB, loss = net.BackwardGradient(x)
    new_A = net.A - diffA*learning_rate
    new_B = net.B - diffB*learning_rate
    net.UpdatePara(new_A, new_B)
    loss = loss.mean()
    Loss.append(loss)
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_ylim([0,0.2])
ax.plot(Loss)

