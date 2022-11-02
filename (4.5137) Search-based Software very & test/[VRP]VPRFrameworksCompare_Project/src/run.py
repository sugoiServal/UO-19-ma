from aco import AcoSolver
from utils import  read_data
#from ortools import OrtoolSolver
import os
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt 

cd = os.getcwd()
parser = argparse.ArgumentParser(description="A CVRP optimizer")
parser.add_argument('--solver', type=str, default='aco', help='search algorithm: "ortool"/ "aco"')    

args = vars(parser.parse_args())


if __name__ == "__main__":
    problems = ['E-n22-k4.vrp', 'E-n23-k3.vrp', 'E-n30-k3.vrp', 'E-n33-k4.vrp', 'E-n51-k5.vrp', 'E-n76-k7.vrp', 'E-n76-k8.vrp', 'E-n76-k10.vrp', 'E-n76-k14.vrp', 'E-n101-k8.vrp', 'E-n101-k14.vrp']    
    #problems = ['E-n22-k4.vrp']
    best_lengths = []
    exe_time = []
    # define the problem file, read as data_in
    #file = r"C:\Users\Boris\Desktop\search_software_eng\project\ga-for-cvrp-master\E-n51-k5.vrp"

    #define the solver
    if args['solver'] == 'ortool':
        solver = OrtoolSolver()
    elif args['solver'] == 'aco':
        solver = AcoSolver()
    
    #solver solve the problem 
    A = 1
    for instance in problems: 
        print("#####################")
        print("%i---------"%A)
        A += 1
        instance_dir = os.path.join('Eilon', instance)
        data = read_data(instance)      
        start = int(round(time.time() * 1000))
        best = solver.solve(data)
        elapse = int(round(time.time() * 1000)) - start
        exe_time.append(elapse)
        if args['solver'] == 'aco':
            best_lengths.append(best[1]) 
    
    

