# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:45:58 2019

@author: Boris
"""

import numpy as np
import matplotlib.pylab as plt
import time
def getData(N, sigma):  
     """
     generate data set
     ------
     N: int, generated data size
     sigma: float, parameter in normal disturbance
     ------
     returns data set [x,y], size = N*2 
     """
     np.random.seed()
     Data = np.zeros([N,2])
     x = np.random.uniform(0, 1, N)
     z = np.random.normal(0, sigma, N)
     y = np.cos(2*np.pi*x) + z
     Data[:,0] = x
     Data[:,1] = y
     return Data
     
def miniBatching(data, batch_size):
    """
    randomly choose batch_size data in data set
    ------
    data:tensor, all data for train
    batch_size:int, #data to be chosen from data
    ------
    returns sub data set [x,y], size = batch_size<N
    """
    idx = np.random.choice(data.shape[0], batch_size)  #for SGD
    data_batch = data[idx]
    return data_batch
    
    
  
def polynomialModel(theta, x):  #batch tensor
     """
     our model/estimator, mapping x to a estimation of target throught polynomial
     -----
     theta: a [#degree, 1]
     x: a [#data, 1]
     -----
     returns a estimate of target, y, a [#data, 1]
     """
     batch_size = x.shape[0]
     degree = theta.shape[0]-1
     X = np.repeat(x, degree, axis=1)
     X = np.power(X, np.arange(1, degree+1))
     X = np.concatenate([np.ones([batch_size,1]), X],axis=1)
     y = X.dot(theta)
     return y
     
def getMSE(y, target):   #scalar
    """
    calculate the MSE of a given estimate and target
    ------
    y: estimate [#data, 1]
    target: target [#data, 1]
    ------
    returns MSE, a scalar
    """

    N = y.shape[0]
    return 1/N*np.sum((y-target)**2)
        

def gradient(theta, x, target, eps = 1e-5, use_regularize= False, reg_penal = 0):
    """
    perform gradient calculation regarding theta
    ------
    theta:[#degree, 1]
    x:[#data, 1]
    target:[#data, 1]
    eps:a small float number
    ------
    returns the average d_loss/d_theta in data set, [#degree, 1]
    """
    data_size = x.shape[0]
    grad =  np.zeros([x.shape[0], theta.shape[0]])   # a matrix of all gradient
    if  use_regularize == True:
        loss = lambda theta: (polynomialModel(theta, x)-target)**2 +  reg_penal*(theta.T.dot(theta).item())
    else:
        loss = lambda theta: (polynomialModel(theta, x)-target)**2
    for idx in range(theta.shape[0]):
        tem = theta[idx].copy()
        theta[idx] = tem+eps
        evalplus = loss(theta)
        theta[idx] = tem-eps
        evalminus = loss(theta)
        theta[idx] = tem
        gradi = (evalplus - evalminus)/(2*eps)  #gradient of theta_i to loss to all data point
        gradi = gradi.reshape(data_size)
        grad[:,idx] = gradi
        
    grad = grad.sum(axis = 0)/data_size    #average gradient
    grad = grad.reshape([theta.shape[0],1])
    return grad
        
        

def fitData(degree, train_size, sigma, test_size = 1000, batch_size=20, learn_rate= 0.25, use_regularize= False, reg_penal = 0, maxiter=10000):          #data:=(#data_pair, 2)
    
    """
    inputs
    degree:int, degree of polynomial model
    train_size: int, training set size
    test_size: int, testing set size
    sigma: float, parameter in normal disturbance
    batch_size: int, size of mini batch
    learn_rate: float, learning rate
    maxiter: int, #loop when optimiazing
    ------
    returns 
    theta: [#degree, 1] the eatimated polynomial coefficients
    Ein: float, the MSE of fitted model over training set
    Eout: float, the MSE of fitted model over test set
    """       
    
    train_data = getData(train_size, sigma)
    theta = np.random.randn(degree+1,1)  
    #loss_history = []
    
    #start_time = time.time()
    for iter in range(maxiter):
        if train_size > 20:
            batch = miniBatching(train_data, batch_size)
        else:
            batch = train_data
        x = batch[:,0]
        x = x.reshape([x.shape[0],1])
        target = batch[:,1]
        target = target.reshape([target.shape[0],1])
        if use_regularize==True:
            grad = gradient(theta, x, target, eps = 1e-6, use_regularize=True, reg_penal=reg_penal)
        else:
            grad = gradient(theta, x, target, eps = 1e-6)

        theta = theta - learn_rate*grad
        #loss = getMSE(polynomialModel(theta, x),target)
        #loss_history.append(loss)
    """    
    #x = train_data[:,0]
    #x = x.reshape([x.shape[0],1])
    #target = train_data[:,1]
    #target = target.reshape([target.shape[0],1])
    #y = polynomialModel(theta, x)
    #plt.scatter(x, target, c='r')
    #xx = np.arange(-1,1,0.01)
    #xx= xx.reshape([xx.shape[0],1])
    #plt.plot(xx, polynomialModel(theta, xx), c='b')   
    """
    #print("--- %s seconds ---" % (time.time() - start_time))  
    
   
    #loss_history = np.array(loss_history)
    #plt.plot(loss_history)
    Ein = getMSE(polynomialModel(theta, x), target)
    #print("Ein is:", Ein)
    
    test_data = getData(test_size, sigma)    
    test_x = test_data[:,0]
    test_x = test_x.reshape([test_x.shape[0],1])
    test_target = test_data[:,1]
    test_target = test_target.reshape([test_target.shape[0],1])
    y = polynomialModel(theta, test_x)
    Eout = getMSE(y, test_target)
    #print("Eout is:", Eout)
    #plt.scatter(test_x, test_target, c='r')
    #plt.scatter(test_x, y, c='b') 
    
    
    return theta, Ein, Eout
    
 
    
def experiment(train_size, degree, sigma, trial=50, test_size = 1000, use_regularize= False, reg_penal = 0):
    """
    do mean calculation:
    ------
    train_size: as N
    degree: as d
    sigma: noise stdanard deviation
    trial: trial times, as M
    ------
    returns:
    Ein_bar
    Eout_bar
    Ebias
    """
    theta_mat = np.zeros([trial, degree+1])
    Eins = np.zeros(trial)
    Eouts = np.zeros(trial)
    for idx in range(trial):
       if use_regularize== True:
           theta, Eins[idx], Eouts[idx] = fitData(degree = degree, train_size = train_size, sigma = sigma, use_regularize=True, reg_penal=reg_penal) 
       else:        
           theta, Eins[idx], Eouts[idx] = fitData(degree = degree, train_size = train_size, sigma = sigma) 
       theta = theta.T
       theta_mat[idx,:] = theta
       #print("loop", idx+1)
    
    Ein_bar = Eins.mean()
    Eout_bar = Eouts.mean()
  
    
    theta_bar = theta_mat.sum(axis=0)/trial
    theta_bar = theta_bar.reshape([degree+1,1])
    test_data = getData(test_size, sigma)    
    test_x = test_data[:,0]
    test_x = test_x.reshape([test_x.shape[0],1])
    test_target = test_data[:,1]
    test_target = test_target.reshape([test_target.shape[0],1])
    y = polynomialModel(theta_bar, test_x)
    Ebias = getMSE(y, test_target)
    
    #plt.scatter(test_x, test_target, c='r')
    #plt.scatter(test_x, y, c='b') 
    
  
    return Ein_bar, Eout_bar, Ebias  
    
    
   

