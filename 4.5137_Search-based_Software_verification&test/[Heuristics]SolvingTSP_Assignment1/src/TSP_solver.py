import argparse
import os
import numpy as np
#import scipy
from matplotlib import pyplot as plt 
cd = os.getcwd()
parser = argparse.ArgumentParser(description="A TSP optimizer")
parser.add_argument('--metaheuristic', type=str, default='HC', help='search algorithm: "HC"/ "SA"')    
parser.add_argument('--file_dir', type=str, default=os.path.join(cd, 'a280.txt'), help='absolute directory to the test file')    

#hill climbing parameters
parser.add_argument('--hc_num_neighbors', default=256, help="how many neighbors each step of climbing choose from")


args = vars(parser.parse_args())



def save_csv(sol):
    return 

class TSPSolver():
    def __init__(self, solver = 'HC'):
        self.solver = solver
            
        return
    def solve(self, file, plotsol=False, save=True):
        data, dmatrix = self.read_data(file)
        print('solving: ' + args['file_dir'] +'\n with ' + self.solver)
        if self.solver == 'HC':
            (best_fit, sol) = self.hill_climbing(data, dmatrix)
        elif self.solver == 'SA':
            (best_fit, sol) = self.simulated_annealing(data, dmatrix)
        else:
            raise ValueError('That was an invaild algorithm')

        print(best_fit)   
        if plotsol:
            self.plot_route(data, sol)
        if save:
            self.save_csv(sol)
            
        return (best_fit, sol)
        
    def save_csv(self, sol):
        np.savetxt(os.path.join(cd, self.solver+'-solution.csv'), (np.array(sol)+1).astype(int), fmt='%d', delimiter=',')        

    def simulated_annealing(self, data, dmatrix):
        t = -81/np.log(0.4)   # average gradient/0.4
        L = int(np.ceil(280*279/2*0.25)) #1/4 poportion of the number of neighbor
        sol = self.generate_random_sol(data)
        def cool(t):
            alpha = 0.95
            return alpha*t
          
        for epoch in range(150):
            print('epoch %d: %f' %(epoch,self.fitness(sol, dmatrix)))
            for step in range(L):
                new_sols, fitness_gradient = self.two_opt(sol, dmatrix, 1)
                if fitness_gradient[0] < 0:
                    sol = new_sols[0]
                else:
                    p = np.exp(-fitness_gradient[0]/t)
                    flag = np.random.binomial(1, p)
                    if flag:
                        sol = new_sols[0]
            t = cool(t)
            best_fit = self.fitness(sol, dmatrix)
        return best_fit, sol
    
        
    def hill_climbing(self, data, dmatrix):
        ## steepest hill climbing
        NUM_NEIGHBOR = args['hc_num_neighbors']       
        climb = True
        sol = self.generate_random_sol(data)
        best_fit = 50000
        while climb:
            climb = False
            new_sols, fitness_gradient = self.two_opt(sol, dmatrix, NUM_NEIGHBOR)
            best = np.argsort(fitness_gradient)[0]
            if fitness_gradient[best] < 0:
                climb = True
                sol = new_sols[best]
                best_fit = self.fitness(sol, dmatrix)
        return best_fit, sol
     
        

    
        
    #init sol, rand permulation of [0,279] in a280 case, list
    def generate_random_sol(self, data):
        return np.random.permutation(len(data)).tolist() 
    
    def read_data(self, file):
        with open(file) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        
        for idx, line in enumerate(content):
            if line == 'NODE_COORD_SECTION':
                start = idx+1
            if line == 'EOF':
                end = idx
        content = content[start: end]
            
                
        data = np.zeros((len(content), 3))        
        for idx, line in enumerate(content):
            content[idx] = line.split()
            for jdx, item in enumerate(content[idx]):
                data[idx][jdx] = item
        loc = data[:, 1:]
        dmatrix = np.sqrt(np.sum((loc[:, np.newaxis, :] - loc[np.newaxis, :, :]) ** 2, axis = -1))
        return data, dmatrix
    
    def two_opt(self, sol, dmatrix, num_neighbor):
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
    
    def fitness(self, sol, dmatrix):
    # take num_node without repetition 
        sol = sol + [sol[0]]   #add_repetition only when calculate fitness
        dist_sum = 0
        for i in range(len(sol)):
            if i == (len(sol)-1):
                dist_sum += dmatrix[sol[i]][sol[0]]
                break 
            dist_sum += dmatrix[sol[i]][sol[i+1]]
        return dist_sum
    def plot_route(self, data, sol):
        ds = data[sol]
        x = ds[:,1]
        y = ds[:,2]
      
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
        #fname = os.path.join(plot_root, str(epoch)+'-'+str(step)+'.png')
        #fig1.show()
        fig1.savefig(os.path.join(cd, self.solver+'-route.png'))


    

if __name__ == "__main__":
    tsp_solver = TSPSolver(solver = args['metaheuristic'])
    (best_fit, sol) = tsp_solver.solve(args['file_dir'], plotsol=False, save=False)

    
        