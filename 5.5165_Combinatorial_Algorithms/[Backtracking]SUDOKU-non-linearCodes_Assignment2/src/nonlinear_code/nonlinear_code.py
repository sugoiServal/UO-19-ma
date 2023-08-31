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

length = 9
min_dist = 4




    

# in solution, we save the codeword's index in V




def greedyColorBounding(A, choose_set):
    # num_color: n
    # cur_color: 0, n-1
    color  =  np.ones(len(choose_set))*-1
    color_class = []  # dynamic size of #color
    num_color = 0
    for i in range(len(choose_set)):
        cur_color = 0
        while (cur_color < num_color):
            if (len(A[i].intersection(color_class[cur_color])) > 0):
                cur_color += 1
            else: 
                break
        if cur_color ==  num_color:
            num_color+=1
            color_class.append(set())
        color_class[cur_color].add(i)
        color[i] = cur_color
    return num_color



def maxNonlinearCode(cur_solution, choose_set):    
    global counter
    global opt_solution
    global opt_size
    counter += 1
    if len(cur_solution) > opt_size:
        opt_size = len(cur_solution)           
        opt_solution = cur_solution.copy()
        
    if len(cur_solution) > 0:
        # last choose set union A(l-1), l-1 is the last element of the current solution
        choose_set =  set(choose_set).intersection(C[cur_solution[-1]])  
    choose_set = sorted(choose_set)             # list

    if len(choose_set) > 0:
        for item in choose_set:
            cur_solution.append(item)
            maxNonlinearCode(cur_solution, choose_set)
    if len(cur_solution)>0:
        cur_solution.pop()
    return 

def maxNonlinearCode_bound(cur_solution, choose_set):    
    global counter
    global opt_solution
    global opt_size
    counter += 1
    if len(cur_solution) > opt_size:
        opt_size = len(cur_solution)
        opt_solution = cur_solution.copy()
        
    if len(cur_solution) > 0:
        # last choose set union A(l-1), l-1 is the last element of the current solution
        choose_set =  set(choose_set).intersection(C[cur_solution[-1]])  
    choose_set = sorted(choose_set)             # list
    # bound
    bound = greedyColorBounding(A, choose_set) + len(cur_solution)
    if len(choose_set) > 0:
        for item in choose_set:
            if bound <= opt_size:
                if len(cur_solution)>0:
                    cur_solution.pop()
                    return
            cur_solution.append(item)
            maxNonlinearCode_bound(cur_solution, choose_set)
    if len(cur_solution)>0:
        cur_solution.pop()
    return 




def printSol(V, opt_solution):
    code_words = np.array(V)[opt_solution]
    print("{", end='')
    for i in range(code_words.shape[0]):
        for j in range(code_words.shape[1]):
            print(code_words[i][j], end='')
        
        if i == code_words.shape[0]-1:
            continue
        print(',', end=' ')   
    print("}\n")
    
    
nodes = []
exec_time = []
opt_sizes = []

# for length in range(4,9):

#     V = generateAllNodes(length)
#     A, B, C = initMatrixAnB(V, min_dist)
    
#     opt_solution = []
#     cur_solution = []  
#     opt_size = 0
#     choose_set = [item for item in range(len(V))]
#     counter = 0   
    
#     start_time = time.time()
#     maxNonlinearCode(cur_solution, choose_set)    
#     running_time = time.time() - start_time
    
#     print("n=%i, d =%i, \nno bounding, execute time: %f, nodes: %i"%(length, 4, running_time, counter))
#     printSol(V, opt_solution)
#     nodes.append(counter)
#     exec_time.append(running_time)
#     opt_sizes.append(opt_size)
    

for length in range(9,10):

    V = generateAllNodes(length)
    A, B, C = initMatrixAnB(V, min_dist)
    
    opt_solution = []
    cur_solution = []  
    opt_size = 0
    choose_set = [item for item in range(len(V))]
    counter = 0   
    
    start_time = time.time()
    maxNonlinearCode_bound(cur_solution, choose_set)  
    running_time = time.time() - start_time
    
    print("n=%i, d =%i, \nwith bounding, execute time: %f, nodes: %i"%(length, 4, running_time, counter))
    printSol(V, opt_solution)
    nodes.append(counter)
    exec_time.append(running_time)
    opt_sizes.append(opt_size)


    