# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:09:00 2019

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
from pandas import DataFrame 






#def __init__(self, device, embedder, input_size=embedding_dim, hidden_size=20, batch_first=True, dropout = 0)

os.chdir(r"C:\Users\funrr\Desktop\dl assignment3")
work_dir = os.getcwd()


train_batch_size = 80
test_batch_size = 150
embedding_dim = 200    #input_size, 50, 300
hidden_dim = [20, 50, 100, 200, 500]  #hidden_size
max_review_length = 2000   #num_layers
learn_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3]

trainloss_tick = 2
testloss_tick = 12
plot_tick = 20
MA_window = 1
#dropout  = 0.15

lridx = 0
hididx= 4
glove = DataClasses.readGloVe()
train_loader, test_loader = DataClasses.imdbDataLoader(train_batch_size = train_batch_size, test_batch_size = test_batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


lstm = RNNclasses.LSTM(device =device, embedder = glove, hidden_size=hidden_dim[hididx]).to(device)

#example = enumerate(train_loader)
#step, (x,y) = next(example)
#out = rnn.forward(x)
optimizer = optim.Adam(lstm.parameters(), lr=learn_rate[lridx], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

#optimizer = optim.SGD(rnn.parameters(), lr=lr, momentum = momentum)    #momentum???

CEloss = nn.BCELoss()
train_acces = DataFrame(columns=['acc'])
test_acces = DataFrame(columns=['acc'])
print(lstm.parameters)

for epoch in range(12):  
        
        for step, (train_data, train_label) in enumerate(train_loader):  #600 steps per epoch
            #train_data = train_data.reshape(train_batch_size, img_length*img_length)
            train_label =train_label.float()
            train_label = train_label.reshape(-1,1)
            train_label =train_label.to(device)            
            optimizer.zero_grad()            
            loss = CEloss(lstm.forward(train_data), train_label)
            if step == 0:
                print("epoch : % 3d,    iter: % 5d,    loss:% .6f" %(epoch+1, step+1, loss.item()))
           
            if (step+1)%trainloss_tick == 0:
                print("epoch : % 3d,    iter: % 5d,    loss:% .6f" %(epoch+1, step+1, loss.item()))
            
            if (step+1)%testloss_tick == 0:
                train_acc = lstm.accuracy(train_data, train_label)
                train_acces =  train_acces.append({'acc':train_acc},ignore_index=True)             
                
                test_sample = enumerate(test_loader)
                tem, (test_data, test_label) = next(test_sample)
                test_label =test_label.float()
                test_label = test_label.reshape(-1,1)
                test_label = test_label.to(device)
                test_loss = CEloss(lstm.forward(test_data), test_label)
                
                print("TESTloss & testAccuracy*******************")
                print("TEST loss:     % .6f" %(test_loss.item()))   
                test_acc = lstm.accuracy(test_data, test_label)
                test_acces = test_acces.append({'acc':test_acc},ignore_index=True)
                print("TRAIN accuracy: % .6f" %(train_acc))
                print("TEST accuracy: % .6f" %(test_acc))
                print("*******************************************")	
            if (step+1)%plot_tick == 0:    
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.plot(test_acces.rolling(window = MA_window).mean(), label='test MA acc')
                ax1.plot(train_acces.rolling(window = MA_window).mean(), label='train MA acc')
                ax1.legend(prop={'size': 9})
                ##############################################################
                title = "LSTM learning curve"+ ', lr='+str(learn_rate[lridx])
                ax1.set_title(title)
                ax1.set_xlabel("3 train steps")
                ax1.set_ylabel("binary CE loss")
                plt.pause(0.05)
                fig1              
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            
fig1.savefig('LSTM ' + str(hidden_dim[hididx]) +'.jpg',dpi = 600)                
#np.save("LTSM test"+str(hidden_dim[hididx]) + "lr= " + str(learn_rate[lridx]), test_acces)
#np.save("LSTM train"+str(hidden_dim[hididx]) + "lr= " + str(learn_rate[lridx]), train_acces)
np.save("LTSM test"+str(hidden_dim[hididx]) , test_acces)
np.save("LSTM train"+str(hidden_dim[hididx]), train_acces)
best = work_dir +'\\best models\\LSTMdim=' + str(hidden_dim[hididx])
torch.save(lstm.state_dict(), best)


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
    test_acc = lstm.accuracy(test_data, test_label)
    scores.append(test_acc)
    torch.cuda.empty_cache()
score = np.mean(scores)
        
            

#######################################################