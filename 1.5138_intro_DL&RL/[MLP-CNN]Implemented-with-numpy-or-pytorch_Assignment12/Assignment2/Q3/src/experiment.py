# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:07:17 2019

@author: Boris
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


"""
Softmax regression:
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
epoches_size = 600
iteration = 30000
lr = 0.06
momentum = 0.5
trainloss_tick = 30
testloss_tick = 120


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_accuracies=[]
for test in range(15):    
    softNet = classes.SoftmaxReg().to(device)

    optimizer = optim.SGD(softNet.parameters(), lr=lr, momentum = momentum) 
    CEloss = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    print(softNet.parameters)
    for epoch in range(20):  #30000/600=50 epoches:int(iteration/epoches_size)
        for step, (train_data, train_label) in enumerate(train_loader):  #600 steps per epoch
            train_data = train_data.reshape(train_batch_size, img_length*img_length)
            train_data, train_label = train_data.to(device), train_label.to(device)         #
            optimizer.zero_grad()            
            loss = CEloss(softNet.forward(train_data), train_label)
            
            if step == 0:                                                           #
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
                        
            if (step+1)%trainloss_tick == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
                
            if (step+1)%testloss_tick == 0:
                train_losses.append(loss.item())
                test_sample = enumerate(test_loader)
                tem, (test_data, test_label) = next(test_sample)
                test_data = test_data.reshape(test_batch_size, img_length*img_length)
                test_data, test_label = test_data.to(device), test_label.to(device)     #
                test_loss = CEloss(softNet.forward(test_data), test_label)
                test_losses.append(test_loss.item())                        #
                print("TESTloss & testAccuracy*******************")
                print("TEST loss:     % .6f" %(test_loss.item()))           #
                           
                test_data = test_all.test_data.reshape(test_all.test_data.shape[0], img_length*img_length)
                test_data = test_data.type(torch.float)
                test_data, test_all.targets = test_data.to(device), test_all.targets.to(device) 
                test_accuracy = softNet.accuracy(test_data, test_all.targets)
                print("TEST accuracy: % .6f" %(test_accuracy))
            
                print("*******************************************")
                
            loss.backward()
            optimizer.step()
    test_data = test_all.test_data.reshape(test_all.test_data.shape[0], img_length*img_length)
    test_data = test_data.type(torch.float)
    test_data, test_all.targets = test_data.to(device), test_all.targets.to(device) 
    test_accuracy = softNet.accuracy(test_data, test_all.targets)
    test_accuracies.append(test_accuracy)       

np.save("Softdrop0BN1_experiment", test_accuracies)
          
                     
            
        
            

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
epoches_size = 600
iteration = 30000
lr = 0.06
momentum = 0.5
trainloss_tick = 30
testloss_tick = 120

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#MLPnn = classes.MLP()
test_accuracies=[]
for test in range(15):    
    structure = np.array([50,30])
    MLPnn = classes.MLP(structure = structure).to(device)
    optimizer = optim.SGD(MLPnn.parameters(), lr=lr, momentum = momentum)    

    CEloss = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    print(MLPnn.parameters)
    for epoch in range(20):  #30000/600=50 epoches:int(iteration/epoches_size)
        for step, (train_data, train_label) in enumerate(train_loader):  #600 steps per epoch
            train_data = train_data.reshape(train_batch_size, img_length*img_length)
            train_data, train_label = train_data.to(device), train_label.to(device) 
            optimizer.zero_grad()            
            loss = CEloss(MLPnn.forward(train_data), train_label)
           
            if step == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
            
            if (step+1)%trainloss_tick == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
           
            if (step+1)%testloss_tick == 0:
                train_losses.append(loss.item())
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
    
      
np.save("MLPdrop0BN1_experiment", test_accuracies)

             
            
 #######################################################       
            

#######################################################

"""
CNN:
    train szie:60000    
    batch size:100
    epoches:600
    iteration:12000
    learning rate:0.1
    
    notes:
       >input will not be flatten
"""
img_length = 28
train_size = 60000  
test_size = 10000  
train_batch_size = 100
test_batch_size = 1000
epoches_size = 600
iteration = 30000
lr = 0.06
momentum = 0.5
trainloss_tick = 30
testloss_tick = 120

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_accuracies=[]
for test in range(15):
    cnn = classes.CNN(conv_kernel=5, pool_kernel=2).to(device)

    optimizer = optim.SGD(cnn.parameters(), lr=lr, momentum = momentum)    #momentum???
    CEloss = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    print(cnn.parameters)
    for epoch in range(20):  #30000/600=20 epoches:int(iteration/epoches_size)
        for step, (train_data, train_label) in enumerate(train_loader):  #600 steps per epoch
            #train_data = train_data.reshape(train_batch_size, img_length*img_length)
            train_data, train_label = train_data.to(device), train_label.to(device)
            optimizer.zero_grad()            
            loss = CEloss(cnn.forward(train_data), train_label)
            
            if step == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
           
            if (step+1)%trainloss_tick == 0:
                print("test: % 2d, epoch : % 3d,    iter: % 5d/600,    loss:% .6f" %(test+1, epoch+1, step+1, loss.item()))
            
            if (step+1)%testloss_tick == 0:
                train_losses.append(loss.item())
                test_sample = enumerate(test_loader)
                tem, (test_data, test_label) = next(test_sample)
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_loss = CEloss(cnn.forward(test_data), test_label)
                test_losses.append(test_loss.item())
                
                print("TESTloss & testAccuracy*******************")
                print("TEST loss:     % .6f" %(test_loss.item()))   
                
                test_data = test_all.test_data.reshape(-1 ,1, img_length, img_length)
                test_data = test_data.type(torch.float)
                test_data, test_all.targets = test_data.to(device), test_all.targets.to(device)
                test_accuracy = cnn.accuracy(test_data, test_all.targets)
                print("TEST accuracy: % .6f" %(test_accuracy))
                print("*******************************************")	
                
            loss.backward()
            optimizer.step()
    test_data = test_all.test_data.reshape(-1 ,1, img_length, img_length)
    test_data = test_data.type(torch.float)
    test_data, test_all.targets = test_data.to(device), test_all.targets.to(device)
    test_accuracy = cnn.accuracy(test_data, test_all.targets)
    test_accuracies.append(test_accuracy)
                
np.save("CNNdrop0BN1_experiment", test_accuracies)
       
        
            