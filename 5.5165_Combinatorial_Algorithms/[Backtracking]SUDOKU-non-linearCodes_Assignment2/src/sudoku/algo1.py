# -*- coding: utf-8 -*-
"""

The first algorithm : try to fill out the first available table position in order (left to right, top to bottom).


"""



import numpy as np
from utils import ValidSol, findChooseSet
import os
import time

#data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'a2data')
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'a2data_new')



"""
row, col index: from 0 to 8, left to right/top to bottom
block index: from 0 to 8, left to right, then top to bottom 
number: from 1


"""



"""# const"""
N = 9


"""# functions"""


        

    
counter = 0
def SUDOKUseqSolve(X, row, col):
    global counter
    N = 9
    if time.time()-st_time > 60*25:
        return None
    #print(counter)
    # reach the end of solution, find solution 
    if row == N-1 and col == N:
        return X
    # reach the end of row, start a new row
    if col == N:
        row += 1
        col = 0
    # the pos has already been filled by the problem, proceed to next position without counting the node
    #print("p2: rpw:%i,col:%i"%(row, col))
    if X[row][col] != 0:
        return SUDOKUseqSolve(X, row, col+1)   #
    
    counter += 1
    # find choose_set, if empty 
    choose_set = findChooseSet(X, row, col)
    # if row == 1  and col == 7:
    #     print(X)
    #     print(choose_set)
    if len(choose_set) > 0:      
        for item in choose_set:
            X[row][col] = item
            sol = SUDOKUseqSolve(X, row, col+1)
            if sol is not None:
                return sol
    # if none in choose_set find a solution, reset the position and backtrack
    X[row][col] = 0
    return None
            

num_nodes = []
exec_times = []
solved = []
problem_name = []


res = []

for file in os.listdir(data_dir):
    if file.endswith(".txt"):                    
        problem = np.genfromtxt(os.path.join(data_dir, file),delimiter=[1]*9, dtype=int)
        if problem.shape[0]>9:    #just some minor inconsistant in the data format 
            problem = problem[:9]
        
        problem_name.append(file)
        counter = 0
        
        st_time = time.time()
        X = problem.copy()
        X = SUDOKUseqSolve(X, 0, 0)
        
        run_time = time.time()-st_time
        exec_times.append(run_time)
        num_nodes.append(counter)
        
        
        print("the problem: %s"%file)
        print(problem)
        print("running time:%d, #nodes:%i"%(run_time, counter))
        if ValidSol(X):
            print("the solution")
            print(X)
            solved.append(1)
        else:
            print("no solution found")
            solved.append(0)
    print("#######################")
    print("\n")



