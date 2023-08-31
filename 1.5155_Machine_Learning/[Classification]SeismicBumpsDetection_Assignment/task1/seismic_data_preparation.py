# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:19:45 2019

@author: Boris
"""

import os
os.chdir(r"C:\Users\Boris\Desktop\machine learning\Assignment_1")
import numpy as np
from pandas import Series,DataFrame
data = np.loadtxt("seismic-bumps.csv", skiprows=154)

data = DataFrame.from_csv("seismic-bumps.csv", sep = ',',header = 151, index_col = None)

for column in ('seismic', 'seismoacoustic', 'ghazard')
for element in ('a', 'b', 'c', 'd'):
    r = (data['seismic'] == element)   
    r*np.ones(2584)
    
    


t = data.values