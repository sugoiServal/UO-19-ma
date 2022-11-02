# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:30:25 2019

@author: funrr
"""



from __future__ import print_function
import DataClasses
import RNNclasses
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pandas import DataFrame 
import pickle






#def __init__(self, device, embedder, input_size=embedding_dim, hidden_size=20, batch_first=True, dropout = 0)

os.chdir(r"C:\Users\funrr\Desktop\dl assignment3")
work_dir = os.getcwd()

train_batch_size = 400
test_batch_size = 1000
embedding_dim = 200    #input_size, 50, 300
hidden_dim = [20, 50, 100, 200, 500]  #hidden_size
max_review_length = 2000   #num_layers
learn_rate = [0.01, 0.05, 0.1, 0.3]

trainloss_tick = 2
testloss_tick = 12
plot_tick = 20
MA_window = 1
#dropout  = 0.15

lridx = 1 #FIX
hididx= 4
glove = DataClasses.readGloVe()
train_loader, test_loader = DataClasses.imdbDataLoader(train_batch_size = train_batch_size, test_batch_size = test_batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################
rnn = RNNclasses.Vanilla(device =device, embedder = glove, hidden_size=hidden_dim[hididx]).to(device)
#example = enumerate(train_loader)
#step, (x,y) = next(example)
#out = rnn.forward(x)
######################################################
optimizer = optim.Adam(rnn.parameters(), lr=learn_rate[lridx], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#optimizer = optim.SGD(rnn.parameters(), lr=lr, momentum = momentum)    

CEloss = nn.BCELoss()
#train_losses = []
#test_losses = []
train_acces = DataFrame(columns=['acc'])
test_acces = DataFrame(columns=['acc'])
print(rnn.parameters)
#for lri in range(3):
for epoch in range(10):  
        
        for step, (train_data, train_label) in enumerate(train_loader):  #600 steps per epoch
            #train_data = train_data.reshape(train_batch_size, img_length*img_length)
            train_label =train_label.float()
            train_label = train_label.reshape(-1,1)
            train_label =train_label.to(device)            
            optimizer.zero_grad()        
            loss = CEloss(rnn.forward(train_data), train_label)
            if step == 0:
                print("epoch : % 3d,    iter: % 5d,    loss:% .6f" %(epoch+1, step+1, loss.item()))
           
            if (step+1)%trainloss_tick == 0:
                print("epoch : % 3d,    iter: % 5d,    loss:% .6f" %(epoch+1, step+1, loss.item()))
                
                #train_losses.append(loss.item())
            if (step+1)%testloss_tick == 0:
                train_acc = rnn.accuracy(train_data, train_label)
                train_acces =  train_acces.append({'acc':train_acc},ignore_index=True)
                
                
                test_sample = enumerate(test_loader)
                tem, (test_data, test_label) = next(test_sample)
                test_label =test_label.float()
                test_label = test_label.reshape(-1,1)
                test_label = test_label.to(device)
                test_loss = CEloss(rnn.forward(test_data), test_label)
                #test_losses.append(test_loss.item())
                
                print("TESTloss & testAccuracy*******************")
                print("TEST loss:     % .6f" %(test_loss.item()))   
                #train_acc = rnn.accuracy(train_data, train_label)
                test_acc = rnn.accuracy(test_data, test_label)
                print("TRAIN accuracy: % .6f" %(train_acc))
                print("TEST accuracy: % .6f" %(test_acc))
                test_acces = test_acces.append({'acc':test_acc},ignore_index=True)

                print("*******************************************")
            if (step+1)%plot_tick == 0:    
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.plot(test_acces.rolling(window = MA_window).mean(), label='test MA acc')
                ax1.plot(train_acces.rolling(window = MA_window).mean(), label='train MA acc')
                ax1.legend(prop={'size': 9})
                ##############################################################
                title = "RNN learning curve"+ ', lr='+str(learn_rate[lridx])
                ax1.set_title(title)
                ax1.set_xlabel("*12 train steps")
                ax1.set_ylabel("binary CE loss")
                plt.pause(0.05)
                fig1
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
#######################################            
fig1.savefig('lr' + str(learn_rate[lridx]) +'.jpg',dpi = 600)                
np.save("RNN test"+str(hidden_dim[hididx]) + "lr= " + str(learn_rate[lridx]), test_acces)
np.save("RNN train"+str(hidden_dim[hididx]) + "lr= " + str(learn_rate[lridx]), train_acces)
best = work_dir +'\\best models\\RNNdim=' + str(hidden_dim[hididx])
torch.save(rnn.state_dict(), best)


#with open(best+str(hidden_dim[hididx])+' glove.npy', 'wb') as handle:
#    pickle.dump(glove, handle, protocol=pickle.HIGHEST_PROTOCOL)

test_loader = DataClasses.DataLoader(DataClasses.imdbData(train = False),\
        batch_size = 1000, shuffle=True)
testor = enumerate(test_loader)
scores = []
for step in range(10):
    step, (test_data, test_label) = next(testor)
    test_label =test_label.float()
    test_label = test_label.reshape(-1,1)
    test_label = test_label.to(device)
    test_acc = rnn.accuracy(test_data, test_label)
    scores.append(test_acc)
    torch.cuda.empty_cache()
score = np.mean(scores)
            

#######################################################