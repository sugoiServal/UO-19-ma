import numpy as np
from math import factorial




def getMobile(direction, permutation):
    max_mobile_abs = 1  #record the abs largest mobile int, init as 1, cannot smaller than 1
    max_mobile_idx = -1

    for i in range(len(permutation)):
        pos_int = abs(permutation[i])      #eg: 3
        dir_int = direction[pos_int-1]  #eg: -1 = left
        next_pos = i + dir_int
        if next_pos >= 0 and next_pos <= n-1:  # should be vaild entry in permutation 
            if abs(permutation[next_pos]) < pos_int: # abs of mobile int should larger than its pointer
                if pos_int > max_mobile_abs:        # find largest mobile int
                    max_mobile_abs = pos_int
                    max_mobile_idx = i                   
    max_mobile = permutation[max_mobile_idx]
    
    
    if max_mobile_abs  == 1:    # if there is no vaild mobile int, then it should be 1 (outer most recur)       
        for i in range(len(permutation)): # get the index of 1
            if abs(permutation[i]) == 1 :
                max_mobile_idx = i
        max_mobile = permutation[max_mobile_idx]
        
    return max_mobile, max_mobile_idx
        

def getMobile_uniTest():
     permutation = [1, -2, 3]
     direction  = [-1, 1, 1] 
     max_mobile, max_mobile_idx = getMobile(direction, permutation)
     # 1, 0
     permutation = [-3, 1, -2]
     direction  = [1, -1, 1] 
     max_mobile, max_mobile_idx = getMobile(direction, permutation)
     #-3, 0
     
     permutation = [2, -1, -3]
     direction  = [1, 1, 1] 
     max_mobile, max_mobile_idx = getMobile(direction, permutation)
     #2, 0
     permutation = [1, 2, -3]
     direction  = [-1, -1, 1] 
     max_mobile, max_mobile_idx = getMobile(direction, permutation)
     #2, 1

def dir2arrow(permutation, direction):
    dirr = []
    for i in range(len(permutation)):
        int_abs = abs(permutation[i])
        if direction[int_abs-1] == -1:
            dirr.append('<')
        else:
            dirr.append('>')
    return dirr


# change n here
n = 1




# position is int-1
neg = -np.zeros(n, dtype=int)
# < is -1 and > is 1, position is int-1
direction  = -np.ones(n, dtype=int)  

permutation = np.arange(1, n+1)





mobile_int = n
result = []
for i in range(2**n*factorial(n)):
    if i % (2*n) == 0:
        print("==========================")
   # print("%d: mobile int %d" %(i, mobile_int))
   
    print(permutation)
    result.append(permutation.tolist())
   # print(dir2arrow(permutation, direction))

    mobile_int, mobile_idx = getMobile(direction, permutation)
    
                

            
    if mobile_int == 1:
        permutation[mobile_idx] = -mobile_int   # negate 1
        direction = -direction                  # flip all direction

        

    else:
        for i in range(abs(mobile_int), n):      # flip direct of bigger int
            direction[i] = -direction[i]    
        if neg[abs(mobile_int)-1] == 1:                    #case of neg action 
                permutation[mobile_idx] = -mobile_int  # negate mobile int and resume neg flag
                neg[abs(mobile_int)-1] = 0
                
    
                
                continue        
        else:  #move and other stuff
                next_pos = mobile_idx + direction[abs(mobile_int)-1] #get the index of the one to be swap
                # swap 
                temp = permutation[mobile_idx]
                permutation[mobile_idx] = permutation[next_pos]
                permutation[next_pos] = temp
    
                
                if next_pos == 0:     #hit left
                    direction[abs(mobile_int)-1] *= -1
                    neg[abs(mobile_int)-1] = 1
    

def print_result(result):
    result = np.array(result)
    result = result.reshape(-1, 2*n, n)
    for i in range (result.shape[0]):
        for j in range(2*n):
            print(result[i].tolist()[j], end = '')
        print("\n")
        
print_result(result) 

            