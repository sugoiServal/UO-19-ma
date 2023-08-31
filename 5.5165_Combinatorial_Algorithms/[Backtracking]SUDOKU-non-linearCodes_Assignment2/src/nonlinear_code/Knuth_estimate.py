# nonliner-code C(n, d): a subset of universe Un
# A(n, d): the size of largest (max) nonliner-code ==> optimization 


# each code word in the universe(n) Un is a node in a big graph
# two nodes are connected only if their dist are greater than d
# we want every pair in C(n, d) to be connected, so its a clique
# thus equal to the maximum clique problem 

# the index of vertex(codeword) start from 0

import numpy as np
from utils import *
import time



def maxNonlinearCodeEstimater(cur_solution, choose_set, s):    
    global N

    if len(cur_solution) > 0:
        # last choose set union A(l-1), l-1 is the last element of the current solution
        choose_set =  set(choose_set).intersection(C[cur_solution[-1]])  
    choose_set = sorted(choose_set)             # list
    
    s =  len(choose_set)*s
    N += s

    if len(choose_set) > 0:
        choice = np.random.randint(len(choose_set))
        cur_solution.append(choose_set[choice])
        maxNonlinearCodeEstimater(cur_solution, choose_set, s)
    else:       # at the end of path
        return 
        
lengths = [str(i) for i in range(4, 12)]        
result = {lengths[i]: [] for i in range(len(lengths))}



for length in range(4, 12):
    min_dist = 4
    V = generateAllNodes(length)
    A, B, C = initMatrixAnB(V, min_dist)
    # in solution, we save the codeword's index in V
  
    trials = [i for i in range(100000,1000000, 200000)]

    start_time = time.time()
    for (idx,trial) in enumerate(trials):
        summation  = 0
        for i in range(trial):    
            s = 1
            N = 1 
            cur_solution = []  
            choose_set = [item for item in range(len(V))]
            maxNonlinearCodeEstimater(cur_solution, choose_set, s)   
            summation += N
        
        running_time = time.time() - start_time
        print("execute time: %f, esetimated nodes(%i trials): %i"%(running_time, trial, summation/trial))
        result[str(length)].append(summation/trial)
        
