# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:04:46 2019

@author: Boris
"""

import numpy as np
from collections import OrderedDict
import os
os.chdir(r"C:\Users\funrr\Desktop\dl assignment 2\Q1")

class Sigmoid:
    def __init__(self):
        self.var_out = None
        self.diff_out = None
    def FP(self, var_in):
        out = 1/(1+np.exp(-var_in))
        self.var_out = out
        return out
    def BP(self, diff_in):
        diff = diff_in *(1.0 - self.var_out) *self.var_out
        self.diff_out = diff
        return diff
class Loss:
    def __init__(self):
        self.var_in = None    #K*N
        self.var_out = None   #1*N
        self.diff_out = None  #K*N
    def FP(self, var_in):      #K*N
        N = var_in.shape[1]
        self.var_in = var_in
        out = np.sum(np.sqrt(var_in*var_in), axis=0)   #a 1*N, not a tuple
        out = out.reshape(1, N)
        self.var_out = out    #1*N
        return out
    def BP(self, diff_in):     #ones(K,N)
        k = diff_in.shape[0]
        var_out = self.var_out.repeat(k, axis=0)
        diff = (self.var_in/var_out)*diff_in
        self.diff_out = diff   #K*N
        return diff

class Product:
    def __init__(self):
        self.var_1 = None
        self.var_1 = None
        self.diff_var1 = None
        self.diff_var2 = None
    def FP(self, var_1, var_2):
        self.var_1 = var_1
        self.var_2 = var_2
        out = var_1*var_2
        return out
    def BP(self, diff_in):
        diff_var1 = self.var_2*diff_in
        diff_var2 = self.var_1*diff_in
        self.diff_var1 = diff_var1
        self.diff_var2 = diff_var2
        return diff_var1, diff_var2

class AffineA:
    def __init__(self, A):
        self.A = A          #/////////////
        self.var_in = None
        self.diff_A = None
        self.diff_var_in = None
    def FP(self, var_in):
        self.var_in = var_in
        out = self.A.dot(var_in)
        return out
    def BP(self, diff_in):
        diffA = diff_in.dot(self.var_in.T)
        diffvar = self.A.T.dot(diff_in)
        self.diff_A = diffA
        self.diff_var_in = diffvar
        return diffA, diffvar
        
class AffineB:
    def __init__(self, B):
        self.B = B           #/////////////k*k
        self.x = None        #k*N
        self.diff_B = None
        self.diff_x = None
    def FP(self, x):
        self.x = x
        out = self.B.dot(x)
        return out
    def BP(self, diff_in):  #k*N
        diffB = diff_in.dot(self.x.T)
        diffx = self.B.T.dot(diff_in)
        self.diff_B = diffB
        self.diff_x = diffx
        return diffB, diffx
        
class network:
    def __init__(self, k): 
        self.A = np.random.randn(k,k)
        self.B = np.random.randn(k,k)
        self.k = k
        self.N = None
        self.FP_trace = OrderedDict()
        self.FP_trace['affA1']=AffineA(self.A)
        self.FP_trace['affB']=AffineB(self.B)
        self.FP_trace['sig']=Sigmoid()
        self.FP_trace['Product']=Product()
        self.FP_trace['affA2']=AffineA(self.A)
        self.FP_trace['affA3']=AffineA(self.A)
        self.FP_trace['Loss']=Loss()
        
        self.BP_trace = OrderedDict()
        
    def ForwardLoss(self, x):        #x:K*N
        if self.N == None:
            self.N = x.shape[1]
        v1 = self.FP_trace['affA1'].FP(x)
        v2 = self.FP_trace['affB'].FP(x)
        v3 = self.FP_trace['sig'].FP(v1)
        v4 = self.FP_trace['Product'].FP(v2,v3)
        v5 = self.FP_trace['affA2'].FP(v4)
        v6 = self.FP_trace['affA3'].FP(v5)
        v7 = self.FP_trace['Loss'].FP(v6)
        return v7
        
    def BackwardGradient(self, x):
        loss = self.ForwardLoss(x)
        one = np.ones([self.k,self.N])
        u1 = self.FP_trace['Loss'].BP(one)
        diffA1, u2 = self.FP_trace['affA3'].BP(u1)
        diffA2, u3 = self.FP_trace['affA2'].BP(u2)
        u4, u5 = self.FP_trace['Product'].BP(u3)   #v2,v3-->B trace first
        diffB, diffx1 = self.FP_trace['affB'].BP(u4)
        u6 = self.FP_trace['sig'].BP(u5)
        diffA3, diffx2 = self.FP_trace['affA1'].BP(u6)
        
        diffA = diffA1 + diffA2 + diffA3
        return diffA, diffB, loss   #loss?
                     
    
    def UpdatePara(self, A, B):
        self.A = A
        self.B = B
        self.FP_trace['affA1'].A = A
        self.FP_trace['affA2'].A = A
        self.FP_trace['affA3'].A = A
        self.FP_trace['affB'].B = B
    