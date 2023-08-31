import argparse
import numpy as np
parser = argparse.ArgumentParser(description="A TSP optimizer")
parser.add_argument('--file_dir', default=20, help="absolute directory to the test file")
parser.add_argument('--metaheuristic', default=20, help="the solver, Hill Climbing/ Tabu search/ Evolutionary/ Simulated annealing")
from scipy import stats



############## here is the function (literally)


#file =  args['file_dir']

file = r'C:\Users\Boris\Desktop\search_software_eng\assignment1\a280.txt'

#return data in numpy (size*3) and distance matrix
def read_data(file):
    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    
    for idx, line in enumerate(content):
        if line == 'NODE_COORD_SECTION':
            content = content[idx+1: -1]
            
    data = np.zeros((len(content), 3))        
    for idx, line in enumerate(content):
            content[idx] = line.split()
            for jdx, item in enumerate(content[idx]):
                data[idx][jdx] = item
    loc = data[:, 1:]
    dmatrix = np.sqrt(np.sum((loc[:, np.newaxis, :] - loc[np.newaxis, :, :]) ** 2, axis = -1))
    return data, dmatrix


#init sol, rand permulation of [0,279] in a280 case, list
def generate_random_sol(data):
    return np.random.permutation(len(data)).tolist() 


def two_opt(sol, dmatrix, num_neighbor):
    num_nodes = dmatrix.shape[0]
    fitness_gradient = []
    new_sols = []
    for i in range(num_neighbor):        
        ## get 2 swap index
        swapin_out_edge = np.random.randint(num_nodes, size = 2)
        while not abs(swapin_out_edge[0]-swapin_out_edge[1])>1:    #cannot be adjacent edge
            swapin_out_edge = np.random.randint(num_nodes, size = 2)
        ## build a new sol
        l_idx = np.min(swapin_out_edge)
        r_idx = np.max(swapin_out_edge)    
        mid = sol[l_idx+1:r_idx+1]
        mid.reverse()
        new_sol = sol[:l_idx+1] + mid + sol[r_idx+1:]
        new_sols.append(new_sol)
        
        ## calculate fitness change 
        if r_idx != (num_nodes-1):
            pre = dmatrix[sol[l_idx]][sol[l_idx+1]] + dmatrix[sol[r_idx]][sol[r_idx+1]]
            after = dmatrix[sol[l_idx]][sol[r_idx]] + dmatrix[sol[l_idx+1]][sol[r_idx+1]]
        else: 
            pre =  dmatrix[sol[l_idx]][sol[l_idx+1]] + dmatrix[sol[r_idx]][sol[0]]
            after = dmatrix[sol[l_idx]][sol[r_idx]] + dmatrix[sol[l_idx+1]][sol[0]]
        fitness_gradient.append(after-pre)
    return new_sols, fitness_gradient

    
    

def fitness(sol, dmatrix):
# take num_node without repetition 
    sol = sol + [sol[0]]   #add_repetition only when calculate fitness
    dist_sum = 0
    for i in range(len(sol)):
        if i == (len(sol)-1):
            dist_sum += dmatrix[sol[i]][sol[0]]
            break 
        dist_sum += dmatrix[sol[i]][sol[i+1]]
    return dist_sum
        


def print_solution(sol):    
    print(sol+1)



        
        
def HillClimbing():
    ## hill climb
    runhist = []
    for run in range(100):
        print(run)
    
        ## steepest hill climbing
        NUM_NEIGHBOR = 256
        #if __name__ == "__main__":
        (data, dmatrix) = read_data(file)
        climb = True
        sol = generate_random_sol(data)
        best_fit = 50000
        while climb:
            climb = False
            new_sols, fitness_gradient = two_opt(sol, dmatrix, NUM_NEIGHBOR)
            best = np.argsort(fitness_gradient)[0]
            if fitness_gradient[best] < 0:
                climb = True
                sol = new_sols[best]
                best_fit = fitness(sol, dmatrix)
        runhist.append(best_fit)      
        
    stats.describe(runhist)


## Simulated Annealing

def cool(t):
    alpha = 0.95
    return alpha*t


def simulated_annealing():
    data, dmatrix = read_data(file)
    t = -81/np.log(0.4)   # average gradient/0.4
    L = int(np.ceil(280*279/2*0.25)) #1/4 poportion of the number of neighbor
    sol = generate_random_sol(data)

   
    #cur_gradient = 1000
    for epoch in range(150):
        print(epoch)
        for step in range(L):
            try:
                new_sols, fitness_gradient = two_opt(sol, dmatrix, 1)
            except:
                print('error')
                return sol
            if fitness_gradient[0] < 0:
                #cur_gradient = fitness_gradient
                sol = new_sols[0]
            else:
                p = np.exp(-fitness_gradient[0]/t)
                flag = np.random.binomial(1, p)
                if flag:
                    sol = new_sols[0]
        t = cool(t)
        print (fitness(sol, dmatrix))

    return sol, fitness(sol, dmatrix)
        

import matplotlib.pyplot as plt
def plot_route(data, sol):
    ds = data[sol]
    x = ds[:,1]
    y = ds[:,2]
  
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
    #fname = os.path.join(plot_root, str(epoch)+'-'+str(step)+'.png')
    fig1.show()
    #.savefig(fname)

    
        
    

