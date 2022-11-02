import math
import random
import numpy
from functools import reduce
import sys
import getopt
import numpy as np


class AcoSolver():
    def __init__(self):
        self.alfa = 2
        self.beta = 5
        self.sigm = 3
        self.ro = 0.8
        self.th = 80
        self.iterations = 20
        self.ants = 22
    
    
    def generateGraph(self, in_data):        
        demands = in_data['demands'].tolist()
        demand ={ i : int(demands[i-1]) for i in range(1, len(demands)+1)} 
        capacityLimit = in_data['capacity']
        optimalValue = in_data['optimal']    
        graph = {i : (int(in_data['coor'][i-1][0]), int(in_data['coor'][i-1][1])) for i in range(1, in_data['dimension']+1)}
        
        vertices = np.arange(2, in_data['dimension']+1).tolist()   
        edges = { (min(a,b),max(a,b)) : numpy.sqrt((graph[a][0]-graph[b][0])**2 + (graph[a][1]-graph[b][1])**2) for a in graph.keys() for b in graph.keys()}
        feromones = { (min(a,b),max(a,b)) : 1 for a in graph.keys() for b in graph.keys() if a!=b }
        
        return vertices, edges, capacityLimit, demand, feromones, optimalValue
    def solutionOfOneAnt(self, vertices, edges, capacityLimit, demand, feromones):
        solution = list()
    
        while(len(vertices)!=0):
            path = list()
            city = numpy.random.choice(vertices)
            capacity = capacityLimit - demand[city]
            path.append(city)
            vertices.remove(city)
            while(len(vertices)!=0):
                probabilities = list(map(lambda x: ((feromones[(min(x,city), max(x,city))])**self.alfa)*((1/edges[(min(x,city), max(x,city))])**self.beta), vertices))
                probabilities = probabilities/numpy.sum(probabilities)
                
                city = numpy.random.choice(vertices, p=probabilities)
                capacity = capacity - demand[city]
    
                if(capacity>0):
                    path.append(city)
                    vertices.remove(city)
                else:
                    break
            solution.append(path)
        return solution
    
    def rateSolution(self, solution, edges):
        s = 0
        for i in solution:
            a = 1
            for j in i:
                b = j
                s = s + edges[(min(a,b), max(a,b))]
                a = b
            b = 1
            s = s + edges[(min(a,b), max(a,b))]
        return s
    
    def updateFeromone(self, feromones, solutions, bestSolution):
        Lavg = reduce(lambda x,y: x+y, (i[1] for i in solutions))/len(solutions)
        feromones = { k : (self.ro + self.th/Lavg)*v for (k,v) in feromones.items() }
        solutions.sort(key = lambda x: x[1])
        if(bestSolution!=None):
            if(solutions[0][1] < bestSolution[1]):
                bestSolution = solutions[0]
            for path in bestSolution[0]:
                for i in range(len(path)-1):
                    feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = self.sigm/bestSolution[1] + feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
        else:
            bestSolution = solutions[0]
        for l in range(self.sigm):
            paths = solutions[l][0]
            L = solutions[l][1]
            for path in paths:
                for i in range(len(path)-1):
                    feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))] = (self.sigm-(l+1)/L**(l+1)) + feromones[(min(path[i],path[i+1]), max(path[i],path[i+1]))]
        return bestSolution
            
    def solve(self, in_data):
        bestSolution = None
        vertices, edges, capacityLimit, demand, feromones, optimalValue = self.generateGraph(in_data)
        for i in range(self.iterations):
            solutions = list()
            for _ in range(self.ants):
                solution = self.solutionOfOneAnt(vertices.copy(), edges, capacityLimit, demand, feromones)
                solutions.append((solution, self.rateSolution(solution, edges)))
            bestSolution = self.updateFeromone(feromones, solutions, bestSolution)
            print(str(i)+":\t"+str(int(bestSolution[1]))+"\t"+str(optimalValue))
        return bestSolution
        
        
        
        








