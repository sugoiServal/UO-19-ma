# -*- coding: utf-8 -*-
"""

The first algorithm : try to fill out the first available table position in order (left to right, top to bottom).


"""



import numpy as np
from utils import *
import time
import os

#data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'a2data')
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'a2data_new')

"""
row, col index: from 0 to 8, left to right/top to bottom
block index: from 0 to 8, left to right, then top to bottom 
number: from 1
depth: from 0 to (#zeros in problem)-1


"""



"""# const"""
N = 9


"""# functions"""    

def initAvailableMap(problem):
    # get the number of blank to be filled(the depth of recursion )

    depth = ((problem ==0)*np.ones_like(problem)).sum()
    
    # to represent the map
    # ==> (depth)*(index(1) + 3 + depth) matrix, column:index,  row, col, block, depth_number_from_[0 to #available number-1[)], one_buffer_colomn; after choosing a number in row/col, entries with the same row/ col/ block -1 available number        
    # uncharted depth are set to default -1
    available_map =  np.ones((depth, depth+3+1))*-1
    
    
    ##build used sets for row, col and block to help initiate available_map  
    row_used =  [{}] * N
    col_used = [{}] * N
    block_used = [{}] * N    
    row_used, col_used, block_used = initUsedsets(problem, row_used, col_used, block_used)
    
    # fill the map by visit all blanks 
    idx = 0
    for row in range(N):
        for col in range(N):
            if problem[row][col] == 0:
                block  = getBlock(row, col) 
                available_map[idx][0] = row
                available_map[idx][1] = col           
                available_map[idx][2] = block
                
                # calculte the inital available number
                used_set = block_used[block].union(col_used[col]).union(row_used[row])
                available_num = 9 -len(used_set)
                available_map[idx][3] = available_num
                idx += 1
    
    # insert a positional index to the first column
    # TODO: maybe useless
    available_map = np.insert(available_map, 0,  np.arange(depth), axis=1)
    return depth, available_map.astype(int)
   
# TODO: careful if available_map being modified as a global safe 

def findNextPos(available_map, cur_depth):
    
    # TODO: careful about depth  
    # search the latest depth and keep for maximum and minimal value(and the position's index), ignore masked item(-1)
    maximum = -1            # just out of range
    minimal = 10            # same
    idx = -1
    for i in range(available_map.shape[0]):
        item = available_map[i, 4+cur_depth]
        if item > maximum and item >= 0:
            maximum = item
        if item < minimal and item > 0:
            minimal = item
            idx = i
                
    # if the biggest available number is zero(maximum), retrun false(dead end)
    if maximum == 0:
        return None    
    # else, retrieve the smallest available number's idx(col, row)
    row = available_map[idx,1]
    col = available_map[idx,2]  
    return idx, row, col

    
def updateAvailableMap(X, cur_depth, row, col):
    global available_map
    
    block = getBlock(row, col)
    for i in range(available_map.shape[0]):
        
        # if the pos have already been masked, the next depth is still masked, override other operation 
        # if the position is just selected, it cannot be chosen further and is masked by -1(by default)
        if available_map[i, 4+cur_depth] == -1 or (available_map[i, 1] == row and available_map[i, 2] == col):
            available_map[i, 4+cur_depth+1] = -1
            continue
                
        # for position that shares the same column /row/ block, recalculate their available number 
        elif available_map[i, 1] == row or available_map[i, 2] == col or available_map[i, 3] == block:
                available_map[i, 4+cur_depth+1] = N - len(getItemUsedset(X, available_map[i, 1], available_map[i, 2]))        
                
        # for other irrelavent positions, just keep the available_map from last depth
        else:
            available_map[i, 4+cur_depth+1] = available_map[i, 4+cur_depth]
        

    

def clearDepth(cur_depth):
    # if the current branch is not feasible, mask the depth so that it is ready for next branch 
    global available_map
    available_map[:, 4+cur_depth] = np.ones(available_map.shape[0])*-1
    pass



counter = 0
#first call SUDOKUdepthSolve(Problem, 0, depth)
def SUDOKUdepthSolve(X, cur_depth, max_depth):
    global available_map
    global counter
    
    if time.time()-st_time > 60*25:
        return None
    counter += 1
    # reach the deepest depth, means a solution found 
    if cur_depth == (max_depth):
        return X

    next_pos = findNextPos(available_map, cur_depth)    ###############   available_map safe here
    if next_pos is None:
        clearDepth(cur_depth)
        return None
    else:
        (idx, row, col) = next_pos
        

    choose_set = findChooseSet(X, row, col)
    
    # print('counter%i, depth%i(%i)'%(counter, cur_depth, cur_depth+4))
    # print(X)
    # print('nexpos: row_%i; col_%i'%(row, col))
    # print(choose_set)

    # print("#########################")
    
    if len(choose_set) > 0:      
        for item in choose_set:
            X[row][col] = item
            updateAvailableMap(X, cur_depth, row, col)   ############### here available_map is modified          
            sol = SUDOKUdepthSolve(X, cur_depth+1, max_depth)
            if sol is not None:
                return sol
    
    clearDepth(cur_depth)
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

        counter = 0
        depth, available_map = initAvailableMap(problem)
        
        X = problem.copy()
        st_time = time.time()
        X = SUDOKUdepthSolve(X, 0, depth)
        
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




        
