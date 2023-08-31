import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools
cwd = os.path.dirname(os.path.realpath(__file__))
cwd = os.path.dirname(cwd) 
os.chdir(cwd)

# # generate data
num_samples = 100000
#num_samples = 1000
num_features = 1000
max_num_cores = 8
# # compile src files 
def compile_src():
    bash_script = """
        cd ./Hogwild_implementation/parallelized_sgd
        make
        chmod +x run
    """
    script_result = os.system(bash_script)
    if script_result != 0:
        print("Script failed...")
    
def datagen(sparsity, ptype):    

    #Enter: <#samples> <#features> <sparsity> <problem type> <filename> <noisy?> <seed>")
    spars_f =  float(sparsity[:1] +'.'+sparsity[1:])
    bash_script = """
        cd ./Hogwild_implementation/data_generation
        python3 data_generation.py %d %d %s %s data-%s.txt
    """ % (num_samples, num_features, spars_f, ptype, sparsity+ptype)
    
    #print(bash_script)
    # Run script and check results
    script_result = os.system(bash_script)
    if script_result != 0:
        print("Script failed...")
    if not os.path.isfile("./Hogwild_implementation/data_generation/data-%s.txt"%(sparsity+ptype)) or not \
            os.path.isfile("./Hogwild_implementation/data_generation/data-%s-iterate.txt"%(sparsity+ptype)):
        print("Data generation failed...")   
    
    bash_script = """
        mv ./Hogwild_implementation/data_generation/data-%s.txt ./Hogwild_implementation/parallelized_sgd
        mv ./Hogwild_implementation/data_generation/data-%s-iterate.txt ./Hogwild_implementation/parallelized_sgd
    """%(sparsity+ptype,sparsity+ptype)

    # Run script and check results
    script_result = os.system(bash_script)
    if script_result != 0:
        print("Script failed...")
    if not os.path.isfile("./Hogwild_implementation/parallelized_sgd/data-%s.txt"%(sparsity+ptype)) or not \
            os.path.isfile("./Hogwild_implementation/parallelized_sgd/data-%s-iterate.txt"%(sparsity+ptype)):
        print("Moving data failed...")

def datadel(sparsity, ptype):    
    bash_script = """
        rm ./Hogwild_implementation/parallelized_sgd/data-%s.txt       
        rm ./Hogwild_implementation/parallelized_sgd/data-%s-iterate.txt
    """%(sparsity+ptype, sparsity+ptype)
    script_result = os.system(bash_script)
    if script_result != 0:
        print("Script failed...")

def cp_iterate(sparsity, ptype, algorithm):
    #this function copy the 'true' label to its corresponding result folder for ploting, and return the target directory and file name
    if ptype == 'r':
        problem = 'linearregression'
    else:
        problem = 'logisticregression'
        
    result_dir = './Hogwild_implementation/parallelized_sgd/%s_%s_data-%s'%(algorithm, problem, sparsity+ptype)
    bash_script = """    
        cp ./Hogwild_implementation/parallelized_sgd/data-%s-iterate.txt %s
    """%(sparsity+ptype, result_dir)
    script_result = os.system(bash_script)
    if script_result != 0:
        print("Script failed...")
    return result_dir, "data-%s-iterate.txt"%(sparsity+ptype)
  
def run(num_cores, sparsity, ptype, algorithm):
# run
    num_total_iterations = 1000000
    step_size = 0.00001
    
    if ptype == 'r':
        problem = 'linearregression'
    else:
        problem = 'logisticregression'
    print("running %s for %s with sparsity %s"%(algorithm, ptype, sparsity))
    
    for num_threads in range(1, num_cores+1):
        # Usage: ./run <num_threads> <data_filename> <problem type> <algorithm name> <total # iterations> <# log points> <stepsize>
        # "Valid problem types: linearregression, logisticregression
        # Valid algorithms: hogwild, exampleindependent, exampleshared, naive, segmentedhogwild
    
        bash_script = """
            cd ./Hogwild_implementation/parallelized_sgd
            ./run %d data-%s.txt %s %s %d 20 %f
        """ % (num_threads, sparsity+ptype, problem, algorithm, num_total_iterations, step_size)
        
        #print(bash_script)
        script_result = os.system(bash_script)
        if script_result != 0:
            print("Run with %d threads failed..." % num_threads) 


        
            
    

def get_iteration_data(num_cores, result_dir):    
    fname = os.path.join(result_dir , "log_%dthreads.csv" % num_cores)
    names = [str(i) for i in range(num_features+1)]
    df = pd.read_csv(fname, header=None, names=names).iloc[1:,:]
    times = df.iloc[:,0].astype(float).values
    df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip().str.lstrip('[').astype(float)
    df.iloc[:, -1] = df.iloc[:, -1].astype(str).str.strip().str.rstrip(']').astype(float)
    iterates = df.iloc[:, 1:].values
    return iterates, times

def l2_norm(v1, v2):
    return sum([(e1-e2)**2 for e1,e2 in zip(v1,v2)])        

def get_sparsity_dirname(sparsity):
    algorithm_name = "naive"
    problem_type = "linearregression"
    regressionclassification_string = "r"
    return "%s_%s_data-%s" % (algorithm_name, problem_type, sparsity+regressionclassification_string)
        
        
def get_execution_time_for_cores(num_cores, sparsity):
    filename = os.path.join('./Hogwild_implementation/parallelized_sgd', get_sparsity_dirname(sparsity) , "threadstats_%dthread.csv" % num_cores)
    real_time = -1.0
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        header_row = next(csv_reader)
        main_row = next(csv_reader)
        real_time = float(main_row[1])
    return real_time

def get_execution_times(maximum_num_cores, sparsity):
    result = []
    for n in range(1, maximum_num_cores):
        result.append(get_execution_time_for_cores(n, sparsity))
    return result    

def plot_convergence(num_cores, result_dir, file, sparsity):
    true_iterate = np.loadtxt(os.path.join(result_dir, file), skiprows = 1)
    plt.clf()
    for num_threads in range(1, num_cores+1, 1):
        iterates, times = get_iteration_data(num_threads, result_dir)
        l2_distances = [l2_norm(iterate, true_iterate) for iterate in iterates]
        plt.plot(times, l2_distances, label="%d thread(s)" % num_threads)
    sparsity_f = float(sparsity) / 10.0
    plt.title("L2 Distance to true weight vs. Time (s), %.1f%% data density"%sparsity_f)
    plt.xlabel("Execution Time (s)")
    plt.ylabel("L2 Distance to true weight")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(os.path.join(result_dir, "%s_convergence.png"%file[:-4]))
        
def plot_time2cores(sparsities):
    plt.figure(figsize=(8,6))
    markers = itertools.cycle(('x', 'o', '^', '+', '.', '*'))
    for sparsity in sparsities:
        exe_time_vs_cores = get_execution_times(max_num_cores, sparsity)
        f_sparsity = float(sparsity) / 10.0
        plt.plot(range(1, max_num_cores), exe_time_vs_cores, label="%.1f%% data density" % f_sparsity, \
                 linewidth=2.5, marker=next(markers), markevery=4)

    plt.xlabel('Number of Cores')
    plt.ylabel('Execution Time (s)')
    plt.yscale('log')
    # plt.title('Execution Time vs. Number of Cores, for varying data densities')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(os.path.join('./Hogwild_implementation/parallelized_sgd/sparsity_naive', "time_naive.png"))
    
def plot_speedup2sparsity(sparsities):
    plt.figure(figsize=(8,6))
    markers = itertools.cycle(('x', 'o', '^', '+', '.', '*'))
    num_cores = range(1, max_num_cores)
    
    for sparsity in sparsities:
        exe_time_vs_cores = get_execution_times(max_num_cores, sparsity)
        speedup_vs_cores = [exe_time_vs_cores[0] / t for t in exe_time_vs_cores]
        f_sparsity = float(sparsity) / 10.0
        plt.plot(num_cores, speedup_vs_cores, label="%.1f%% data density" % f_sparsity, \
                 linewidth=2.5, marker=next(markers), markevery=4)
    
    plt.plot(num_cores, num_cores, label="Theoretical Upper Bound", linestyle='dashed')
    plt.plot(num_cores, np.ones(len(num_cores)), label="Theoretical Lower Bound", linestyle='dashed')
        
    plt.xlabel('Number of Cores')
    plt.ylabel('Speedup')
    # plt.yscale('log')
    plt.ylim(0,10)
    # plt.title('Speedup vs. Number of Cores, for varying data densities')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(os.path.join('./Hogwild_implementation/parallelized_sgd/sparsity_naive', "speedup_naive.png"))


def main():
    compile_src()
    sparsities = ['0005', '0010', '0100', '0200' , '0500', '1000']
    ptypes = ['r', 'c']
    algorithms = ['hogwild', 'naive']
    for sparsity in sparsities:
        for ptype in ptypes:     
            datagen(sparsity, ptype) 
            for algorithm in algorithms:
                run(max_num_cores, sparsity, ptype, algorithm)                        
                result_dir, file = cp_iterate(sparsity, ptype, algorithm)
                plot_convergence(max_num_cores, result_dir, file, sparsity)                
            datadel(sparsity, ptype)
    plot_time2cores(sparsities)  
    plot_speedup2sparsity(sparsities)    


if __name__ == "__main__":
    main()      
  




