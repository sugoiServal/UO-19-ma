# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:25:59 2019

@author: Boris
"""
import DataClasses
import os
from shutil import copyfile
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import csv
from string import punctuation
import time

os.chdir(r"C:\Users\funrr\Desktop\dl assignment3")
work_dir = os.getcwd()

train_batch_size = 10
test_batch_size = 10
embedding_dim = 200    #input_size
hidden_dim = [20, 50, 100, 200, 500]  #hidden_size
max_review_length = 2000   #num_layers
#dropout  = 0.15



class Vanilla(nn.Module):              
        
        def __init__(self, device, embedder, input_size=embedding_dim, hidden_size=20, batch_first=True, dropout = 0):  #hidden size to be examine
            super(Vanilla, self).__init__()  
            self.embedder = embedder
            self.device = device
            self.input_size = input_size
            self.hidden_size = hidden_size
            #self.feature = nn.Sequential(
            self.rnn = nn.RNN(input_size = input_size,\
                    hidden_size=hidden_size, \
                    num_layers =1)
            self.linear = nn.Linear(hidden_size, 1, bias= True)        
                    
                    
                    
                    #nn.Conv2d(in_channels=1, out_channels=10, kernel_size =conv_kernel),
                    #nn.ReLU(),
                    #nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
                    
                                
                
        def EmbeddingPadding(self, x):
            #in: a tuple, reviews(string) of batch_size     
            #out:embedded and padded, [max_len_in_batch(some were padeded), batch_size, embedding_dim]          
            
            review_len = []          
            review_vec = []
            for review in x:
                review_vect = DataClasses.sentenceTransform(review)
                review_len.append(len(review_vect))
                review_vect = DataClasses.wordEmbedding(review_vect, self.embedder)
                review_vect = torch.from_numpy(review_vect)
                review_vec.append(review_vect)
            #seq_len = max(review_len)
            review_vec = torch.nn.utils.rnn.pad_sequence(review_vec, batch_first=True)
            review_vec = review_vec.float()

            review_vec = review_vec.to(self.device)

            pack = torch.nn.utils.rnn.pack_padded_sequence(review_vec,lengths=review_len, batch_first=True,  enforce_sorted=False)

            return pack
        
        def forward(self, x):    #x: STRING IN BATCH!!!, TUPLE
            batch_size = len(x)
            h0 = torch.randn(1, batch_size, self.hidden_size).to(self.device)
            x = self.EmbeddingPadding(x)   # CONSTRUCTION match nn dimension
            #pack = torch.nn.utils.rnn.pack_padded_sequence(x,lengths=review_len, batch_first=True,  enforce_sorted=False)
            output, hn = self.rnn(x, h0)         # output: shape (seq_len, batch, hidden_size)         
            out, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            #mean_out = out.mean(dim=1)       #shape[batch, hidden_size]
            #out = self.linear(mean_out)
            out = torch.sigmoid(self.linear(out.mean(dim=1)))  #[0,1], batch_size*1
            
            return out   
                       
        def accuracy(self, x, label, out=None):  #need mod
            if out is None:
                out = self.forward(x)
            out[out>=0.5] = 1
            out[out<0.5] = 0
            hit = torch.sum((out == label)).item()
            return hit/label.shape[0]
            
class LSTM(nn.Module):              
        
        def __init__(self, device, embedder, input_size=embedding_dim, hidden_size=20, batch_first=True, dropout = 0):  #hidden size to be examine
            super(LSTM, self).__init__()  
            self.embedder = embedder
            self.device = device
            self.input_size = input_size
            self.hidden_size = hidden_size
            #self.feature = nn.Sequential(
            self.lstm = nn.LSTM(input_size = input_size,\
                    hidden_size=hidden_size, \
                    num_layers =1)
            self.linear = nn.Linear(hidden_size, 1, bias= True)        
                    
                    
                    

                    
                                
                
        def EmbeddingPadding(self, x):
            #in: a tuple, reviews(string) of batch_size     
            #out:embedded and padded, [max_len_in_batch(some were padeded), batch_size, embedding_dim]          
            
            review_len = []          
            review_vec = []
            for review in x:
                review_vect = DataClasses.sentenceTransform(review)
                review_len.append(len(review_vect))
                review_vect = DataClasses.wordEmbedding(review_vect, self.embedder)
                review_vect = torch.from_numpy(review_vect)
                review_vec.append(review_vect)
            #seq_len = max(review_len)
            review_vec = torch.nn.utils.rnn.pad_sequence(review_vec, batch_first=True)
            review_vec = review_vec.float()

            review_vec = review_vec.to(self.device)

            pack = torch.nn.utils.rnn.pack_padded_sequence(review_vec,lengths=review_len, batch_first=True,  enforce_sorted=False)

            return pack
        
        def forward(self, x):    #x: STRING IN BATCH!!!, TUPLE
            batch_size = len(x)
            h0 = torch.randn(1, batch_size, self.hidden_size).to(self.device)
            c0 = torch.randn(1, batch_size, self.hidden_size).to(self.device)
            x = self.EmbeddingPadding(x)   # CONSTRUCTION match nn dimension
            #pack = torch.nn.utils.rnn.pack_padded_sequence(x,lengths=review_len, batch_first=True,  enforce_sorted=False)
            output, (hn,cn) = self.lstm(x, (h0, c0))         # output: shape (seq_len, batch, hidden_size)         
            out, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            #mean_out = out.mean(dim=1)       #shape[batch, hidden_size]
            #out = self.linear(mean_out)
            out = torch.sigmoid(self.linear(out.mean(dim=1)))  #[0,1], batch_size*1
            
            return out   
                       
        def accuracy(self, x, label):  #need mod
            
            out = self.forward(x)
            out[out>=0.5] = 1
            out[out<0.5] = 0
            hit = torch.sum((out == label)).item()
            return hit/label.shape[0]            



