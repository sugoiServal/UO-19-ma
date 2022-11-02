# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 17:42:57 2019

@author: Boris
"""
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
from torch.utils.data import Dataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
import csv
from string import punctuation
import time

os.chdir(r"C:\Users\funrr\Desktop\dl assignment3")
work_dir = os.getcwd()


embedding_dim = 200
train_batch_size = 10
test_batch_size = 10
max_review_length = 2000


def readGloVe():
    print("Loading Glove Model")
    gloveFile = work_dir +'\\'+ 'glove.6B.200d.txt'
    model = {}
    with open(gloveFile,encoding='UTF-8') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("200d Done.")
    return model


"""
def EmbeddingPadding(x, glove):
            #in: a tuple, reviews(string) of batch_size     
            #out:embedded and padded, [max_len_in_batch(some were padeded), batch_size, embedding_dim]          
            review_len = []          
            review_vec = []
            for review in x:
                review_vect = sentenceTransform(review)
                review_len.append(len(review_vect))
                review_vect = wordEmbedding(review_vect, glove)
                review_vect = torch.from_numpy(review_vect)
                review_vec.append(review_vect)
            #seq_len = max(review_len)
            review_vec = torch.nn.utils.rnn.pad_sequence(review_vec, batch_first=True)
            review_vec = review_vec.float()
            pack = torch.nn.utils.rnn.pack_padded_sequence(review_vec,lengths=review_len, batch_first=True,  enforce_sorted=False)
            return pack
"""       
def getWordVec(word, words):
    #given a word, find its GloVe vector in GloVe
    if word in words:
        return words[word]
    else:
        words[word] = np.random.rand(1,embedding_dim)
       
        return words[word]
def sentenceTransform(sentenceList):
    #gives a string sentenceList, return a cleaned, tokenized sentenceList
    sentenceList = sentenceList.lower()      #Tolowercase
    soup = BeautifulSoup(sentenceList, features="lxml")          #eliminate HLTMmarker
    sentenceList = soup.get_text()     
    sentenceList = tokenize.wordpunct_tokenize(sentenceList)   #tokenize     
    sentenceList = [c for c in sentenceList if c not in punctuation]          
    #stop_words = set(stopwords.words('english'))
    #sentenceList = [w for w in sentenceList if not w in stop_words]  
                                         #eliminate stop word?
    return sentenceList

def wordEmbedding(sentenceList, words):
    #gives a tokenized sentenceList, return a set of word vectors
    vec_sentence = np.zeros([len(sentenceList),embedding_dim])
    for idx, word in enumerate(sentenceList):
        vec_sentence[idx,:] = getWordVec(word, words).reshape(1,-1)
    return vec_sentence

def dataCopy():
    #store training data and test data (w/label) in 2 DataFrame, and save it to a csv file
    os.makedirs(work_dir+'\\dataset')
    newdata_dir = work_dir+'\\dataset'
    train_data = pd.DataFrame(columns=['comment', 'sentiment'])               #2B saved in newdata_dir
    test_data = pd.DataFrame(columns=['comment', 'sentiment'])      #2B saved in newdata_dir
    socdata_dir = work_dir+'\\imdb-movie-reviews-dataset'   
    idx = 0
    for root, dirs, files in os.walk(socdata_dir):
         for filename in files:           
            if re.match(".*\d+_\d+.txt", filename):
                idx +=1
                print(filename+ str(idx))
                filedir = root + '\\' + filename
                if 'test' in root:    
                    if root[-3:] =='pos': #getlabel for file
                       label = 1  
                    elif root[-3:] =='neg':
                       label = 0
                    else: continue        #skip supervise
                    with open(filedir,encoding='UTF-8') as f:  #get comment
                           comment =f.readlines()
                    test_data = test_data.append({'comment': comment[0],'sentiment':label},ignore_index=True)

           
                elif  'train' in root:
                    if root[-3:] =='pos': #getlabel for file
                       label = 1  
                    elif root[-3:] =='neg':
                       label = 0
                    else: continue        #skip supervise
                    with open(filedir,encoding='UTF-8') as f:  #get comment
                           comment =f.readlines()
                    train_data = train_data.append({'comment': comment[0],'sentiment':label},ignore_index=True)
    traindir = newdata_dir+'\\train_data.csv'
    testdir = newdata_dir+'\\test_data.csv'

    train_data.to_csv(traindir, index=False, encoding='utf8')
    test_data.to_csv(testdir ,index=False, encoding='utf8')

def dataVisual():  #plot histogram of review length
    newdata_dir = work_dir+'\\dataset'
    traindata = pd.read_csv(newdata_dir+'\\train_data.csv')
    testdata = pd.read_csv(newdata_dir+'\\test_data.csv')
    train_reviews_len = []
    test_reviews_len = []
    for idx  in range(len(traindata.index)):
        train_reviews_len.append(len(traindata.iloc[idx][0]))
    for idx  in range(len(testdata.index)):
        test_reviews_len.append(len(testdata.iloc[idx][0]))
    pd.Series(train_reviews_len).hist()
    
    pd.Series(test_reviews_len).hist()
    pd.Series(train_reviews_len).plot.box()
    pd.Series(test_reviews_len).plot.box()
    print("maximum value in training set:" + str(max(train_reviews_len)))
    print("minimal value in training set:" + str(min(train_reviews_len)))
    print("maximum value in test set:" + str(max(test_reviews_len)))
    print("minimal value in test set:" + str(min(test_reviews_len)))
    


def dataDrop():
    #keep only reviews of length from 500 to 2k
    newdata_dir = work_dir+'\\dataset'
    traindata = pd.read_csv(newdata_dir+'\\train_data.csv')
    testdata = pd.read_csv(newdata_dir+'\\test_data.csv')
    trainout = traindata
    testout = testdata

    for idx in  range(len(traindata.index)):
        if len(traindata.iloc[idx][0]) > max_review_length or len(traindata.iloc[idx][0]) < 500:
              trainout = trainout.drop(index = idx)
    trainout = trainout.reset_index()
    trainout = trainout.drop(columns = 'index')
    print('training set truncated, size = ' + str(len(trainout.index)))


    for idx in  range(len(testdata.index)):
        if len(testdata.iloc[idx][0]) > max_review_length or len(testdata.iloc[idx][0]) < 500:
            testout = testout.drop(index = idx)
    testout = testout.reset_index()
    testout = testout.drop(columns = 'index')
    print('test set truncated, size = ' + str(len(testout.index)))

    
    newdata_dir = work_dir+'\\dataset'
    traindir = newdata_dir+'\\train_data_truncate.csv'
    testdir = newdata_dir+'\\test_data_truncate.csv'
    trainout.to_csv(traindir, index=False, encoding='utf8')
    testout.to_csv(testdir ,index=False, encoding='utf8')
    

    
    

    
class imdbData(Dataset):
    #return vectorized sentence groups through indexing (tensor); used by imdbDataLoader
    
    def __init__(self, train=True):
        self.newdata_dir = work_dir+'\\dataset'
        self.train = train   
        if self.train is True:
            self.data = pd.read_csv(self.newdata_dir +'\\train_data_truncate.csv')
        else:
            self.data = pd.read_csv(self.newdata_dir +'\\test_data_truncate.csv')
        #self.words = readGloVe()
    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        comment = self.data.iloc[idx][0]
        label = self.data.iloc[idx][1]
        
        #comment = sentenceTransform(comment)
        #comment_v = wordEmbedding(comment, self.words)
        #comment_v = torch.from_numpy(comment_v)
        #pad = nn.ZeroPad2d((0,0,0,max_review_length-comment_v.shape[0]))
        #pad(comment_v)

        return comment, label


def imdbDataLoader(train_batch_size = 10, test_batch_size=10):
    train = imdbData(train = True)
    print('training set size:' + str(len(train)))
    train_loader = DataLoader(imdbData(train = True),\
        batch_size = train_batch_size, shuffle=True)
    
    test = imdbData(train = False)
    print('test set size:' + str(len(test)))
    test_loader = DataLoader(imdbData(train = False),\
        batch_size = test_batch_size, shuffle=True)
    return train_loader, test_loader
