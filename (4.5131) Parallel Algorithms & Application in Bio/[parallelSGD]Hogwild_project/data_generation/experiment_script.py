import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def get_iteration_data(num_cores):
    fname = "./Hogwild_implementation/parallelized_sgd/hogwild_linearregression_data/log_%dthreads.csv" % num_cores
    names = [str(i) for i in range(num_features+1)]
    df = pd.read_csv(fname, header=None, names=names).iloc[1:,:]
    times = df.iloc[:,0].astype(float).values
    df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip().str.lstrip('[').astype(float)
    df.iloc[:, -1] = df.iloc[:, -1].astype(str).str.strip().str.rstrip(']').astype(float)
    iterates = df.iloc[:, 1:].values
    return iterates, times

def l2_norm(v1, v2):
    return sum([(e1-e2)**2 for e1,e2 in zip(v1,v2)])












num_samples = 100000
num_features = 1000
sparsity = 0.1 # between 0.0 and 1.0 [1,3,5,7, 9, 10]

# AUTO data generation, covers logistic/ linear/ different sparsity /noise


#num_collision

bash_script = """
    cd ./Hogwild_implementation/data_generation
    python3 data_generation.py %d %d %f r data.txt
""" % (num_samples, num_features, sparsity)

#("Enter: <#samples> <#features> <sparsity> <problem type> <filename> <noisy?> <seed>")
# Run script and check results
script_result = os.system(bash_script)
if script_result != 0:
    print("Script failed...")
if not os.path.isfile("./Hogwild_implementation/data_generation/data.txt") or not \
       os.path.isfile("./Hogwild_implementation/data_generation/data-iterate.txt"):
    print("Data generation failed...")
    
    
    
# make file 





# execute the computaion, should be fast, store

num_cores in range(19):

for num_threads in range(1, num_cores+1):
    bash_script = """
        cd ./Hogwild_implementation/parallelized_sgd
        ./run %d data.txt linearregression hogwild %d 20 %f
    """ % (num_threads, num_total_iterations, step_size)
    script_result = os.system(bash_script)
    if script_result != 0:
        print("Run with %d threads failed..." % num_threads)

# retrieve the result and plot, compare 



rm the data_file 


