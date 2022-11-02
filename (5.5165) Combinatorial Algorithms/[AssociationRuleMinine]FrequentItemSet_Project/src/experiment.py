from ECLAT import ECLAT
from apriori import Apriori
from FP import FPtree
import csv
import os
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dpt = os.getcwd()
data_path1 = os.path.join(dpt, "data\Groceries_dataset.csv")
data_path2 = os.path.join(dpt, "data\data2.txt")
def readData1(data_path):
    """
    Read the datasee, generate all transaction by grouping member and data;  
    get a list of all items and a mapping from item to integer
    !get rid of repetitive items in transactions
    - Input:
        - data path
    -return:
        - hrz_data: list of list, horizontal transaction database 
        - items_to_idx: mapping from item name to integer
        - idx_to_items: mapping from integer name to item name

    """
    # read data
    if not os.path.isfile(data_path): raise ValueError('Invalid data path.')
    groceries = pd.read_csv(data_path)
    hrz_data = [transaction[1]['itemDescription'].tolist() for transaction in list(groceries.groupby(['Member_number', 'Date']))]
    
    # get commodity set
    items = set()
    for i in range(len(hrz_data)):
        tran_goods = set(hrz_data[i]) 
        items = items.union(tran_goods)
    items = sorted(list(items))
    items_to_idx = {items[i]: i for i in range(len(items))}
    idx_to_items = {i: items[i] for i in range(len(items))}
    
    # transform the database
    output = [[] for tid in range(len(hrz_data))]
    for tid, transaction in enumerate(hrz_data):
        for item in range(len(transaction)):          
            output[tid].append(items_to_idx[transaction[item]]) 
        output[tid] = sorted(list(set(output[tid])))  
          
    return  output, items_to_idx, idx_to_items
    
def readData2(data_path):
    if not os.path.isfile(data_path): raise ValueError('Invalid data path.')
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        f = csv.reader(f, delimiter=' ', quotechar='\r')
        for row in f:
            data.append([int(item) for item in row])

    return data   

data_name = ['d1', 'd2']
minsups = [i/100 for i in [5, 2, 1.5, 1, 0.75, 0.5]]
algorithms = ['ECLAT', 'FPtree', 'Apriori']

result = {algo:[[],[]] for algo in algorithms}


for algo in algorithms:
    for didx, data in enumerate(data_name):
        # load data 
        if data == 'd1':            
            data_path = data_path1
            (hrz_data, _, _) = readData1(data_path)
        else:
            data_path = data_path2
            hrz_data = readData2(data_path)
        
        minsu_res = []    
        for minsup in minsups:            
            st_time = time.time()
            if algo == "ECLAT":     
                ecl = ECLAT(hrz_data, minsup) 
                sol = ecl.solve()
            elif algo == "FPtree": 
                fp = FPtree(hrz_data, minsup)
                sol = fp.solve()
            elif algo == "Apriori":
                ap = Apriori(hrz_data, minsup)
                sol = ap.solve()
            run_time = time.time() - st_time 
            
            minsu_res.append(run_time)
        
        result[algo][didx] =  minsu_res
            

# linestyle='dashed'
algorithms = ['ECLAT', 'FPtree', 'Apriori']
data_name = ['datasetA', 'datasetB']
for idx, dataset in enumerate(data_name):
    img_dir = r"C:\Users\Boris\Desktop\itemsetmining\PROJ\Apriori-and-Eclat-Frequent-Itemset-Mining-master"
    plt.figure(figsize=(8,6))
    plt.clf()
    for algo in algorithms:
        plt.plot(result[algo][idx], label=algo)   

            
    plt.xticks([0, 1, 2, 3, 4, 5], ['5%', '2%', '1.5%', '1%', '0.75%', '0.5%'])   
    plt.xlabel('minsup')
    plt.ylabel('execution time(s)')
    plt.title(dataset + ' execution time compare')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(img_dir, dataset+".png"),dpi=500)

algorithms = ['ECLAT', 'FPtree']    
data_name = ['datasetA', 'datasetB']
for idx, dataset in enumerate(data_name):
    img_dir = r"C:\Users\Boris\Desktop\itemsetmining\PROJ\Apriori-and-Eclat-Frequent-Itemset-Mining-master"
    plt.figure(figsize=(8,6))
    plt.clf()
    for algo in algorithms:
        plt.plot(result[algo][idx], label=algo)   

            
    plt.xticks([0, 1, 2, 3, 4, 5], ['5%', '2%', '1.5%', '1%', '0.75%', '0.5%'])   
    plt.xlabel('minsup')
    plt.ylabel('execution time(s)')
    plt.title(dataset + ' execution time compare (w/o Apriori)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(img_dir, dataset+"_wo.png"),dpi=500)

