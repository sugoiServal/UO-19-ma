# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:36:54 2019

@author: funrr
"""

import os
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

os.chdir(r"C:\Users\Boris\Desktop\machine learning\assignment 2\ML2")
work_dir = os.getcwd()





def ReadData():
    work_dir = os.getcwd()
    col = '@attribute seismic {a,b,c,d} @attribute seismoacoustic {a,b,c,d} @attribute shift {W, N} @attribute genergy real @attribute gpuls real'\
        + '@attribute gdenergy real @attribute gdpuls real @attribute ghazard {a,b,c,d} @attribute nbumps real @attribute nbumps2 real @attribute nbumps3 real'\
        + '@attribute nbumps4 real @attribute nbumps5 real @attribute nbumps6 real @attribute nbumps7 real @attribute nbumps89 real @attribute energy real'\
        + '@attribute maxenergy real @attribute class {1,0}'
    col = word_tokenize(col)    
    headers = []
    for idx, item in enumerate(col):
        if item == 'attribute':
            headers.append(col[idx + 1])
    data = pd.read_csv(work_dir +'\\seismic_bumps.csv', header = None, names=headers)
    return data
    
    
    

def CheckMissing(data):
    #data[data.isnull().any(axis=1)]  #OR use
    missing_tag = 0
    for (columnName, columnData) in data.iteritems():    
        nullidx = data.index[data.isnull()[columnName] == True].tolist()
        if len(nullidx) != 0: 
            print('Missing Colunm Name : ', columnName)
            print('Missing idx: ')
            print(*nullidx, sep = ", ")  
            missing_tag = 1

    if missing_tag == 0:
        print('No missing value found')
        
def DataDistVis(data):
    work_dir = os.getcwd()
    plotdir = work_dir + "\\plots\\"
    os.makedirs(plotdir)
    
    for idx, col in enumerate(data.columns):
        ax = data[col].hist() 
        fig = ax.get_figure()
        fig.savefig(plotdir + str(idx) + '.' + col +'.jpg', dpi = 600)
        plt.clf()
        
def Cat2Num(data):
    cat_nums = {"seismic":     {"a": 0, "b": 1, "c": 2, "d": 3},
                "seismoacoustic": {"a": 0, "b": 1, "c": 2, "d": 3},
                "shift":     {"W": 0, "N": 1},
                "ghazard":     {"a": 0, "b": 1, "c": 2, "d": 3}}
    data.replace(cat_nums, inplace=True)
    data = data.astype({'seismic': 'int64'}) #why
    data = data.astype({'seismoacoustic': 'int64'})
    data = data.astype({'ghazard': 'int64'})
    return data

