import numpy as np




def ValidSol(X):
    N = 9
    if X is None or X.min() == 0:
        return False
    ##used set by row, col and block    
    row_used =  [{}] * N
    col_used = [{}] * N
    block_used = [{}] * N
    row_used, col_used, block_used = initUsedsets(X, row_used, col_used, block_used)
    for i in range(9):
        if len(row_used[i]) < 9 or len(col_used[i])< 9 or len(block_used[i])< 9:
            return False
    return True


def initUsedsets(problem, row_used, col_used, block_used):
    # """ init feasible set container from problem, used in algo2 to choose next pos """
    
    # rows and columns
    for row in range(9):
        row_used[row]=  {item for item in np.unique(problem[row, :]) if item!=0}
    for col in range(9):
        col_used[col]=  {item for item in np.unique(problem[:, col]) if item!=0}       
    # blocks
    for i in range(3):
        for j in range(3):            
            block_used[i*3+j] = {item for item in np.unique(problem[i*3:(i+1)*3, j*3:(j+1)*3]) if item!=0}
    return row_used, col_used, block_used



def getBlock(row, col):
    # give row and column, return the item's block index (0-8)
    r =  (row - row%3)/3
    c =  (col - col%3)/3
    return int((r*3 + c))

def getItemUsedset(X, row, col):
    # get a given position's usedset 
    row_used =  {item for item in np.unique(X[row, :]) if item!=0}
    col_used =  {item for item in np.unique(X[:, col]) if item!=0}    
    
    block_i = row - row%3
    block_j = col - col%3
    block_used = {item for item in np.unique(X[block_i:block_i+3, block_j:block_j+3]) if item!=0}
    return row_used.union(col_used).union(block_used)

def findChooseSet(X, row, col):
   
    # find choose set given the current partial solution X and position(row+col) 
    # the choose set is: [1, 2, ..., 9] union [row_unused] union [col_unused] union [block_unused]
        
    N = 9
    pi = {i for i in range(1, N+1)}
    
    used_set = getItemUsedset(X, row, col)
    
    choose_set = pi - used_set
    return choose_set 
