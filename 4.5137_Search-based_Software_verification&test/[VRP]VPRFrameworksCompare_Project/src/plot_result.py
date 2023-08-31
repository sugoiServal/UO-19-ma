import numpy as np
import matplotlib.pyplot as plt
import itertools

name = ['aco', 'ortools', 'savings', 'sweep', '2phase']

time = [10867,11399,16998,19873,40358,80217,79783,79929,82371,133933,133621,2019,2020,2003,2002,2004,2010,2003,2051,2003,2250,2296,1,1,3,3,8,18,18,20,20,32,36,13,14,25,27,65,145,143,153,160,251,269,6,37,15,31,26,111,66,27,15,185,65]
length = [383.841,588.9946,530.7621,890.5382,647.8034,847.3993,916.4818,1009.507,1186.277,1086.813,1358.977,367,558,531,850,568,718,757,905,1189,883,1158,388.7723,638.0601,534.4482,843.0978,584.048,746.5285,790.166,878.1822,1078.778,886.4566,1148.379,454.3231,599.05,825.6811,1240.685,812.8325,1131.768,1182.016,1285.956,1436.621,1420.252,1672.26,375.2798,600.6765,608.5463,855.5523,628.2649,711.5751,761.54,994.1885,1172.88,871.7441,1135.21]
best = [375,569,534,835,521,682,735,830,1021,817,1071]

time = np.reshape(np.array(time), (5,11))
time = np.log(time)/np.log(2)
length = np.reshape(np.array(length), (5,11))

result_time = {name[i] : time[i] for i in range(5)}
result_length = {name[i] : length[i] for i in range(5)}

problems = ['n22-k4', 'n23-k3', 'n30-k3', 'n33-k4', 'n51-k5', 'n76-k7', 'n76-k8', 'n76-k10', 'n76-k14', 'n101-k8', 'n101-k14']    

plt.figure(figsize=(9,7))
markers = itertools.cycle(('x', 'o', '^', '+', '.', '*'))

for key in result_time.keys():
    plt.plot(problems, result_time[key], label=key, \
             linewidth=2.5, marker=next(markers), markevery=4)
        
plt.title("Solvers' execution time in each problem, measured in log2(ms)")
plt.xlabel('problem')
plt.ylabel('exec time (ms) (log2)')
plt.ylim(0,22)
plt.legend()
plt.grid()
plt.savefig(r".\time.png")



plt.figure(figsize=(9,7))
markers = itertools.cycle(('x', 'o', '^', '+', '.', '*'))

for key in result_length.keys():
    plt.plot(problems, result_length[key], label=key, \
             linewidth=2.5, marker=next(markers), markevery=4)
plt.plot(problems, best, label="optimal value")    

plt.title("Solvers' optimization result")
plt.xlabel('problem')
plt.ylabel('fitness')
plt.legend()
plt.grid()
plt.savefig(r".\cost.png")


plt.figure(figsize=(9,7))
markers = itertools.cycle(('x', 'o', '^', '+', '.', '*'))

for key in result_length.keys():
    #if key != 'ortools':
        plt.plot(result_time[key], result_length[key], label=key, \
                 linewidth=2.5, marker=next(markers), markevery=4)


plt.title("performance-time trade-off ")
plt.xlabel('exec time (ms) (log2)')
plt.ylabel('fitness')
plt.legend()
plt.grid()
plt.savefig(r".\tradeoff.png")