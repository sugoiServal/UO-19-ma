import os 
import numpy as np
import time
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'a3graphs/a3graphs')



def readFile(data_dir):
    # data: which of the two vertices are connected, skip header 1.
    
    problems = {}
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):   
            print(file)                 
        problem = np.genfromtxt(os.path.join(data_dir, file),skip_header = 1, dtype=int)
        problems[file] = problem
    
    return problems


def problemAdjMat(problem):
    # Input: raw data\, which of the two vertices are connected, skip header 1.
    # output: adjacency matrix of same problem 
    # vertice to self: 1 or 0???
    (unique, _) = np.unique(problem.flatten(), return_counts=True)  # 3518 edges, 100 vertices
    v_size  = unique.size
    adj_mat =  np.zeros((v_size, v_size))
    for i in range(problem.shape[0]):
        edge = problem[i]
        adj_mat[edge[0]][edge[1]] = 1
    #print("there are %i vertices and %i edges"%(v_size, adj_mat.sum()))
    
    # make the matrix symmetric
    i_lower = np.tril_indices(v_size, -1)
    adj_mat[i_lower] = adj_mat.T[i_lower] 
    return adj_mat



def initAdjSet(adj_mat):
    AS = [set() for _ in range(adj_mat.shape[0])]
    vertices_number = adj_mat.shape[0]
    for i in range(vertices_number):
        for j in range(vertices_number):
            if adj_mat[i][j] == 1:
                AS[i].add(j)
    return AS



def isClique(sol, adj_mat):
    # given the adj_mat and a solution (vertex set), check if it is clique
    # get a sub adj_mat of sol.size(), then count the number of 1
    sol = list(sol)
    sub_mat = np.eye(adj_mat.shape[0], dtype= int)
    sub_mat = adj_mat[np.ix_(sol, sol)] 
    for i in range(sub_mat.shape[0]):
        sub_mat[i][i]  = 1
    if int(sub_mat.sum()) < sub_mat.shape[0]**2:
        return False
    else :
        return True
    

def hillClimbing_greedy(adj_mat, runs):
    # random k initialization, improved until can't improve anymore      
    AS = initAdjSet(adj_mat)
    result_l = []
    sol_best = set()
    l_best = 0
    for run_idx in range(runs):        
        v = np.random.randint(0, adj_mat.shape[0])
        sol = {v}
        l = 1
        choose_set = AS[v].copy()
        while len(choose_set) > 0:
            #print(len(choose_set))
            Y_best = -1
            best_size = -1
            for i in choose_set:
                cur_size = len(choose_set.intersection(AS[i]))
                if cur_size > best_size:
                    Y_best = i
                     
            sol.add(Y_best)
            l += 1
            choose_set = choose_set.intersection(AS[Y_best]).copy()
        if l > l_best:
            l_best = l
            sol_best = sol.copy()
        result_l.append(l)
    l_avg = np.array(result_l).mean()   
    return (l_best, sol_best, l_avg)

def hillClimbing_random(adj_mat, runs):
    # random k initialization, improved until can't improve anymore      
    AS = initAdjSet(adj_mat)
    result_l = []
    sol_best = set()
    l_best = 0
    for run_idx in range(runs):        
        v = np.random.randint(0, adj_mat.shape[0])
        sol = {v}
        l = 1
        choose_set = AS[v].copy()
        while len(choose_set) > 0:
            #print(len(choose_set))
            y = np.random.choice(tuple(choose_set))
                     
            sol.add(y)
            l += 1
            choose_set = choose_set.intersection(AS[y]).copy()
        if l > l_best:
            l_best = l
            sol_best = sol.copy()
        result_l.append(l)
    l_avg = np.array(result_l).mean()  
    return (l_best, sol_best, l_avg)




def simulatedAnnealing(adj_mat, T0, alpha, max_iterations):
    # initial tem T0;
    # alpha cooling factor
    
    V = {i for i in range(adj_mat.shape[0])}
    AS = initAdjSet(adj_mat)
    v = np.random.randint(0, adj_mat.shape[0])
    sol = {v}
    sol_best = set()
    l = 1
    l_best = 0
    choose_set = AS[v].copy()
    T = T0
    c = 0
    for c in range(max_iterations):
        r = np.random.random()
        if r < np.exp(-2/T) or len(choose_set) == 0 or len(choose_set)+len(sol) <= l_best:
            if len(sol) > 0:
                v = np.random.choice(tuple(sol))
                sol.remove(v)
                choose_set = V.copy()
                for v in sol:
                    choose_set =  choose_set.intersection(AS[v])
                l -= 1
            else:
                continue
        else:
            y = np.random.choice(tuple(choose_set))
                     
            sol.add(y)
            l += 1
            choose_set = choose_set.intersection(AS[y]).copy() 
            if l > l_best:
                l_best = l
                sol_best = sol.copy()
        T = T*alpha
    
    return (l_best, sol_best)


########## EXPERIMENT SCRIPT    
# all 50000 runs 
# result [problem_idx, best_opt, average_opt, time]
problems = readFile(data_dir)
problem_names = list(problems.keys())
problem_idx = {problem_names[i]: i for i in range(6)}

SA_results = {"random-climb": [[] for i in range(6)]}


## HC 
HC_results  = np.zeros((2, 6, 4)) # two algorithm variation, 6 problem, result*4
i = 0
for name in problem_names:
    problem  = problems[name]        
    adj_mat = problemAdjMat(problem)
    
    st = time.time()
    (l_best, sol_best, l_avg) = hillClimbing_random(adj_mat, 10000)
    rt = time.time() - st

    HC_results[0][i] = [problem_idx[name], l_best, l_avg, rt]
    print(name)   
    print("l_best %i, l_avg %i, time %f"  %(l_best, l_avg, rt))
    print("###############################\n")
    
    
    st = time.time()
    (l_best, sol_best, l_avg) = hillClimbing_greedy(adj_mat, 10000)
    rt = time.time() - st

    HC_results[1][i] = [problem_idx[name], l_best, l_avg, rt]
    print(name)   
    print("l_best %i, l_avg %i, time %f"  %(l_best, l_avg, rt))
    print("###############################\n")
    
    i+=1
    
## SA 
T0 = [5, 20, 50]
A = [0.9, 0.98, 0.995]
result = {}
for t0 in T0:
    for alpha in A:
        print([t0, alpha])
        result[str([t0, alpha])] = np.zeros((6, 4))
        i = 0
        for name in problem_names:
            print(name)
            problem  = problems[name]        
            adj_mat = problemAdjMat(problem)
            
            ls = []
            rts = []
            for run in range(50):
                st = time.time()
                (l_best, sol_best) = simulatedAnnealing(adj_mat, T0 = t0, alpha=alpha, max_iterations=10000)
                rt = time.time() - st
                ls.append(l_best)
                rts.append(rt)
            result[str([t0, alpha])][i] = [problem_idx[name], np.array(ls).max(), np.array(ls).mean() , np.array(rts).mean() ]
            

            i += 1
        
    
    
    



    

