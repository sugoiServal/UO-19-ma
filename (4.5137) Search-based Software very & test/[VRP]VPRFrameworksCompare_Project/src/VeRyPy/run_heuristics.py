#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
""" This file is a part of the VeRyPy classical vehicle routing problem
heuristic library and demonstrates the simple use case of solving a single
TSPLIB formatted problem instance file with a single heuristic algorithm and
printing the resulting solution route by route."""
###############################################################################

import cvrp_io
import os
import time
from classic_heuristics.parallel_savings import parallel_savings_init

from classic_heuristics.sweep import sweep_init

from classic_heuristics.cmt_2phase import cmt_2phase_init

from classic_heuristics.lr3opt import lr3opt_init



from util import sol2routes
from utils import read_data

def calLength(tours, instance):
    demetrix = read_data(instance)["dmatrix"]
    total_length = 0
    for i in range(len(tours)):
        for node in range(len(tours[i])-1):
            total_length += demetrix[tours[i][node]][tours[i][node+1]]
    return total_length






problems = ['E-n22-k4.vrp', 'E-n23-k3.vrp', 'E-n30-k3.vrp', 'E-n33-k4.vrp', 'E-n51-k5.vrp', 'E-n76-k7.vrp', 'E-n76-k8.vrp', 'E-n76-k10.vrp', 'E-n76-k14.vrp', 'E-n101-k8.vrp', 'E-n101-k14.vrp']
best_lengths = []
exe_time = []
for instance in problems:
    print("execute " + instance)
    instance_dir = os.path.join('Eilon', instance)
    problem = cvrp_io.read_TSPLIB_CVRP(instance_dir)

    '''ps : Clarke & Wright (1964) parallel savings algorithm.
    
    CW64-PS:Clarke, G. and Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. Operations Research, 12(4):568-581.'''
    start = int(round(time.time() * 1000))
    solution = parallel_savings_init(
        D=problem.distance_matrix, 
        d=problem.customer_demands, 
        C=problem.capacity_constraint)
    elapse = int(round(time.time() * 1000)) - start
    tours = sol2routes(solution)
    total_length = calLength(tours, instance_dir)
        
    for route_idx, route in enumerate(sol2routes(solution)):
        print("Route #%d : %s"%(route_idx+1, route))
    print("\ntotal length: %d"%(total_length))
    print("elapse %d"%(elapse))
    
    best_lengths.append(total_length)
    exe_time.append(elapse)
    #################################################
    '''swp : Sweep algorithm without any route improvement heuristics.
    
    GM74-SwRI: Gillett, B. E. and Miller, L. R. (1974). A heuristic algorithm for the vehicle-dispatch problem. Operations Research, 22(2):340-349.'''
    start = int(round(time.time() * 1000))
    solution = sweep_init(
        coordinates = problem.coordinate_points,
        D=problem.distance_matrix, 
        d=problem.customer_demands, 
        C=problem.capacity_constraint)
    elapse = int(round(time.time() * 1000)) - start
    tours = sol2routes(solution)
    total_length = calLength(tours, instance_dir)
        
    for route_idx, route in enumerate(sol2routes(solution)):
        print("Route #%d : %s"%(route_idx+1, route))
    print("\ntotal length: %d"%(total_length))
    print("elapse %d"%(elapse))
    best_lengths.append(total_length)
    exe_time.append(elapse)
    ####################################################
    '''cmt : Christofides, Mingozzi & Toth (1979) two phase heuristic.'''
    start = int(round(time.time() * 1000))
    solution = cmt_2phase_init(
        D=problem.distance_matrix, 
        d=problem.customer_demands, 
        C=problem.capacity_constraint)
    elapse = int(round(time.time() * 1000)) - start
    tours = sol2routes(solution)
    total_length = calLength(tours, instance_dir)
        
    for route_idx, route in enumerate(sol2routes(solution)):
        print("Route #%d : %s"%(route_idx+1, route))
    print("\ntotal length: %d"%(total_length))
    print("elapse %d"%(elapse))
    best_lengths.append(total_length)
    exe_time.append(elapse)
