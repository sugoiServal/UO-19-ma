# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:24:01 2019

@author: funrr
"""
from __future__ import print_function
import classes
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



os.chdir(r"C:\Users\funrr\Desktop\dl assignment 2\Q3")
data_dir = os.getcwd()




train_loader, test_loader, train_all, test_all = classes.MNISTLoader()



###########################################################################

            

#######################################################


###########################################################################


"""
MLP:
    train szie:60000    
    batch size:100
    epoches:600
    iteration:12000
    learning rate:0.1
    
    notes:
       >input will be flatten
"""

img_length = 28
train_size = 60000  
test_size = 10000  
train_batch_size = 100
test_batch_size = 1000
iteration = 30000
lr = 0.06
momentum = 0.5
trainloss_tick = 30
testloss_tick = 120

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#MLPnn = classes.MLP()
test_accuracies=[]
train_losses_result = np.zeros((15,700))
test_losses_result = np.zeros((15,175))
for test in range(15):    
    structure = np.array([2048,2048,2048,2048,2048,1024,1024])
    MLPnn = classes.MLP(structure = structure).to(device)
    optimizer = optim.SGD(MLPnn.parameters(), lr=lr, momentum = momentum)    

    CEloss = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    print(MLPnn.parameters)
    for epoch in range(35):  #30000/600=50 epoches:int(iteration/epoches_size)
        for step, (train_data, train_label) in enumerate(train_loader):  #600 steps per epoch
            train_data = train_data.reshape(train_batch_size, img_length*img_length)
            train_data, train_label = train_data.to(device), train_label.to(device) 
            optimizer.zero_grad()            
            loss = CEloss(MLPnn.forward(train_data), train_label)
           
            if step == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
                
            if (step+1)%trainloss_tick == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
                train_losses.append(loss.item())
            if (step+1)%testloss_tick == 0:
                test_sample = enumerate(test_loader)
                tem, (test_data, test_label) = next(test_sample)
                test_data = test_data.reshape(test_batch_size, img_length*img_length)
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_loss = CEloss(MLPnn.forward(test_data), test_label)
                test_losses.append(test_loss.item()) 
                print("TESTloss & testAccuracy*******************")
                print("TEST loss:     % .6f" %(test_loss.item())) 
                test_data = test_all.test_data.reshape(test_all.test_data.shape[0], img_length*img_length)
                test_data = test_data.type(torch.float)
                test_data, test_all.targets = test_data.to(device), test_all.targets.to(device)
                test_accuracy = MLPnn.accuracy(test_data, test_all.targets)
                print("TEST accuracy: % .6f" %(test_accuracy))
                
                print("*******************************************")
                
            loss.backward()
            optimizer.step()
    test_data = test_all.test_data.reshape(test_all.test_data.shape[0], img_length*img_length)
    test_data = test_data.type(torch.float)
    test_data, test_all.targets = test_data.to(device), test_all.targets.to(device)
    test_accuracy = MLPnn.accuracy(test_data, test_all.targets)
    test_accuracies.append(test_accuracy)
    train_losses_result[test, :] = train_losses
    test_losses_result[test, :] = test_losses
    
      
np.save("MLPdrop_1_DEEP_experiment(1)", test_accuracies)
np.save("MLPdrop_1_DEEP_experiment(2)", train_losses_result)
np.save("MLPdrop_1_DEEP_experiment(3)", test_losses_result)


##################################################################
"""
MLP:
    train szie:60000    
    batch size:100
    epoches:600
    iteration:12000
    learning rate:0.1
    
    notes:
       >input will be flatten
"""

img_length = 28
train_size = 60000  
test_size = 10000  
train_batch_size = 100
test_batch_size = 1000
iteration = 30000
lr = 0.06
momentum = 0.5
trainloss_tick = 30
testloss_tick = 120

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#MLPnn = classes.MLP()
test_accuracies=[]
train_losses_result = np.zeros((15,700))
test_losses_result = np.zeros((15,175))
for test in range(35):    
    structure = np.array([12288])
    MLPnn = classes.MLP(structure = structure).to(device)
    optimizer = optim.SGD(MLPnn.parameters(), lr=lr, momentum = momentum)    

    CEloss = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    print(MLPnn.parameters)
    for epoch in range(35):  
        for step, (train_data, train_label) in enumerate(train_loader):  #600 steps per epoch
            train_data = train_data.reshape(train_batch_size, img_length*img_length)
            train_data, train_label = train_data.to(device), train_label.to(device) 
            optimizer.zero_grad()            
            loss = CEloss(MLPnn.forward(train_data), train_label)
           
            if step == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
                
            if (step+1)%trainloss_tick == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
                train_losses.append(loss.item())
            if (step+1)%testloss_tick == 0:
                test_sample = enumerate(test_loader)
                tem, (test_data, test_label) = next(test_sample)
                test_data = test_data.reshape(test_batch_size, img_length*img_length)
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_loss = CEloss(MLPnn.forward(test_data), test_label)
                test_losses.append(test_loss.item()) 
                print("TESTloss & testAccuracy*******************")
                print("TEST loss:     % .6f" %(test_loss.item())) 
                test_data = test_all.test_data.reshape(test_all.test_data.shape[0], img_length*img_length)
                test_data = test_data.type(torch.float)
                test_data, test_all.targets = test_data.to(device), test_all.targets.to(device)
                test_accuracy = MLPnn.accuracy(test_data, test_all.targets)
                print("TEST accuracy: % .6f" %(test_accuracy))
                
                print("*******************************************")
                
            loss.backward()
            optimizer.step()
    test_data = test_all.test_data.reshape(test_all.test_data.shape[0], img_length*img_length)
    test_data = test_data.type(torch.float)
    test_data, test_all.targets = test_data.to(device), test_all.targets.to(device)
    test_accuracy = MLPnn.accuracy(test_data, test_all.targets)
    test_accuracies.append(test_accuracy)
    train_losses_result[test, :] = train_losses
    test_losses_result[test, :] = test_losses
    
      
np.save("MLPdrop_1_BREAD_experiment(1)", test_accuracies)
np.save("MLPdrop_1_BREAD_experiment(2)", train_losses_result)
np.save("MLPdrop_1_BREAD_experiment(3)", test_losses_result)             
            


