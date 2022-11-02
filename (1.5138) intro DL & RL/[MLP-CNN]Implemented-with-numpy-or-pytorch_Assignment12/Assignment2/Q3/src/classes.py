# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:51:42 2019

@author: Boris Li
"""

from __future__ import print_function
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.chdir(r"C:\Users\funrr\Desktop\dl assignment 2\Q3")
data_dir = os.getcwd()

img_hight= 28
img_size = 784
train_batch_size = 100
test_batch_size = 1000
train_size = 60000
test_size = 10000
learning_rate = 0.1
iteration = 10000

def MNISTLoader():
    train_loader = DataLoader(\
        torchvision.datasets.MNIST(data_dir,\
            train=True, \
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Normalize((0.1,),(0.5,))])),\
        batch_size = train_batch_size, shuffle=True)
    train_all = torchvision.datasets.MNIST(data_dir,\
            train=True, \
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Normalize((0.1,),(0.5,))]))
  
    test_loader = DataLoader(\
        torchvision.datasets.MNIST(data_dir,\
            train=False, \
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Normalize((0.1,),(0.5,))])),\
        batch_size = test_batch_size, shuffle=True)
    test_all = torchvision.datasets.MNIST(data_dir,\
            train=False, \
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Normalize((0.1,),(0.5,))]))
    return train_loader, test_loader, train_all, test_all
    
"""
def ReadData():
        files = {'train_img':'train-images-idx3-ubyte.gz', 'train_label':'train-labels-\
        idx1-ubyte.gz','test_img':'t10k-images-idx3-ubyte.gz','test_label': 't10k-labels-idx1-ubyte.gz'}
        file_path = data_dir + "\\" + files['train_img']
        f = gzip.open(file_path, 'rb')
        train_img = np.frombuffer(f.read(), np.uint8, offset=16)
        train_img = train_img.reshape(int(train_img.shape[0]/(28**2)), 28, 28) 
        
        file_path = data_dir + "\\" + files['test_img']
        f = gzip.open(file_path, 'rb')
        test_img = np.frombuffer(f.read(), np.uint8, offset=16)
        test_img = test_img.reshape(int(test_img.shape[0]/(28**2)), 28, 28) 
        
        file_path = data_dir + "\\" + files['train_label']
        f = gzip.open(file_path, 'rb')
        train_label = np.frombuffer(f.read(), np.uint8, offset=8)
        
        file_path = data_dir + "\\" + files['test_label']
        f = gzip.open(file_path, 'rb')
        test_label = np.frombuffer(f.read(), np.uint8, offset=8)
        
        torch_train = torch.from_numpy(train_img)
        torch_train = torch_train.type(torch.float)
        torch_test = torch.from_numpy(test_img)
        torch_test = torch_test.type(torch.float)
        
        torch_trainlabel = torch.from_numpy(train_label)     
        torch_testlabel = torch.from_numpy(test_label)
        torch_trainlabel =torch_trainlabel.type(torch.int64)
        #Onehot_trainlabel= F.one_hot(torch_trainlabel, num_classes = 10) #60000*10
        torch_testlabel =torch_testlabel.type(torch.int64)
        #Onehot_testlabel= F.one_hot(torch_testlabel, num_classes = 10)  #10000*10
        
        
        return train_img, test_img, train_label, test_label
"""


def FlattenImg(img_batch):   #N*C*28*28 -> N* (C*784)
        N = img_batch.shape[0]
        img_batch = img_batch.reshape(N, -1)
        return img_batch
            
       
    
class SoftmaxReg(nn.Module):
        def __init__(self):
            super(SoftmaxReg, self).__init__()
            self.bn = nn.BatchNorm1d(784)  #bn
            self.linear = nn.Linear(img_size, 10, bias= False)
            
        
        def forward(self, x):    #x: tensor:N*784, batch_size=N
            x = self.bn(x)   #bn
            out  = self.linear(x)
            #out = F.dropout(out)   #do
            out = F.softmax(out)
            return out    #N*10: one_hot
        def accuracy(self, x, label):
            out = self.forward(x)
            out = torch.argmax(out, dim=1)
            hit = torch.sum((out == label)).item()
            return hit/label.shape[0]
      
class MLP(nn.Module):               
        
        #structure:array[#node in Hiddenlayer1, #node in Hiddenlayer2,...]:
        
        def __init__(self, structure = None):
            super(MLP, self).__init__()
            if structure is None:
                self.fw = nn.Sequential(
                    nn.Linear(img_size, 50),
                    nn.ReLU(),
                    nn.Linear(50, 30),
                    #nn.Dropout(),   #do
                    #nn.BatchNorm1d(30),  #bn
                    nn.ReLU(),
                    nn.Linear(30, 10))
        
            else:
                structure_dict = OrderedDict()
                for i in range(structure.size):
                    if i == 0:
                        structure_dict["affine1"]=nn.Linear(img_size, structure[0])
                        structure_dict["ReLU1"] = nn.ReLU()
                        
                    else:
                        label0 = "dropout"+ str(i+1)
                        label1 = "affine" + str(i+1)
                        label2 = "ReLU" + str(i+1)
                        structure_dict[label0]=nn.Dropout(p=0.15)
                        structure_dict[label1]=nn.Linear(structure[i-1], structure[i])
                        structure_dict[label2] = nn.ReLU()
                        
                       
                label = "affine" + str(structure.size+1)
                structure_dict[label] = nn.Linear(structure[structure.size-1], 10)              
                
                self.fw = nn.Sequential(structure_dict)
                 
        def forward(self, x):    #x: tensor:N*784, batch_size=N
            x = self.fw(x)
            out = F.softmax(x)
            return out    #N*10: one_hot
            
        def accuracy(self, x, label):
            out = self.forward(x)
            out = torch.argmax(out, dim=1)
            hit = torch.sum((out == label)).item()
            return hit/label.shape[0]


class CNN(nn.Module):              
        
        def __init__(self, conv_kernel=5, pool_kernel=2):
            super(CNN, self).__init__()
            pool_stride = pool_kernel
            self.feature = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=10, kernel_size =conv_kernel),
                    nn.BatchNorm2d(10), #bn
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
                    
                    
                    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=conv_kernel),
                    nn.BatchNorm2d(20),  #bn
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
            self.fc = nn.Sequential(
                    #nn.Dropout2d(),    #do
                    nn.Linear(4*4*20, 200),
                    nn.BatchNorm1d(200),  #bn
                    nn.ReLU(),
                    nn.Linear(200, 10))                
                    
            
        
        def forward(self, x):    #x: N*1*28*28
            x = self.feature(x)
            x = FlattenImg(x)
            x = self.fc(x)
            out = F.softmax(x) 
            return out   
            
            
        def accuracy(self, x, label):
            out = self.forward(x)
            out = torch.argmax(out, dim=1)
            hit = torch.sum((out == label)).item()
            return hit/label.shape[0]
        
