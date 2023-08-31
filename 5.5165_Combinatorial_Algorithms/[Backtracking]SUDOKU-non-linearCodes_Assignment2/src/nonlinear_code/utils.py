#import numpy as np
def generateAllNodes(length):
    # generate all binary codeword of length (length), order in binary number increasing order
    string = [[] for _ in range(length)] 
    V = [[] for _ in range(2**length)]
    
    V, number = generateAllBinary(length, string, 0, V, 0) 
    return V
    
    
def generateAllBinary(length, string, i, V, number):
    if i == length:
        V[number] = string.copy()
        number += 1
        return V, number
    
    string[i] = 0
    (V, number) = generateAllBinary(length, string, i+1, V, number)    
    string[i] = 1
    (V, number) = generateAllBinary(length, string, i+1, V, number)
    return V, number
    
    
def hammingDist(x, y):
    if len(x) != len(y):
        raise ValueError("x and y length not consistent")
    dist = 0
    for idx in range(len(y)):
        if x[idx] != y[idx]:
            dist += 1
    return dist
                
    

    
    
def initMatrixAnB(V, d):
    # A: adjacency matrix, ie: neighborhood whose hamming distance >= d
    # B: set of nodes whose index is greater than v in binary number increasing order
    # since A and B are both static, we calculate their union directly and return as C
    A = [set() for _ in range(len(V))]        # save index of codeword
    B = [set() for _ in range(len(V))]        # save index of codeword
    C = [set() for _ in range(len(V))]        # save index of codeword
    
    # B
    for v in range(len(V)):
        for neighbor in range(v+1,len(V)):
            B[v].add(neighbor)
    # C       
    for v in range(len(V)):
        for neighbor in B[v]:
            if hammingDist(V[v], V[neighbor]) >= d:
                C[v].add(neighbor)    
    
    # A       
    for v in range(len(V)):
        for neighbor in range(len(V)):
            if hammingDist(V[v], V[neighbor]) >= d:
                A[v].add(neighbor)  
    
    return A, B, C
        

def printSol(V, optSol):
    for i in range():
        pass
    
    
    
