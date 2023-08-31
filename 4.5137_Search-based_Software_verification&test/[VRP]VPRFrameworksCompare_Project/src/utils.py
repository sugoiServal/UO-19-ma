import numpy as np 
file  = r'C:\Users\Boris\Desktop\search_software_eng\project\ga-for-cvrp-master\E-n51-k5.vrp'

def read_data(file):
     with open(file) as f:
         content = f.readlines()
     content = [x.strip() for x in content] 
     optimalValue = None    
     for idx, line in enumerate(content):
         if line[:3] == "COM":
             optimalValue = int(content[idx].split()[-1][:-1])
         if line[:3] == "DIM":
             dimension = int(content[idx].split()[-1])
         if line[:3] == "CAP":
             capacity = int(content[idx].split()[-1])
         if line == 'NODE_COORD_SECTION':
             coor_start = idx+1
         if line == 'DEMAND_SECTION':
             coor_end = idx
         if line == 'DEMAND_SECTION':
             demand_start = idx+1
         if line == 'DEPOT_SECTION':
             demand_end = idx  
     coor_str = content[coor_start: coor_end]
     denamds_str = content[demand_start: demand_end]
     
     coor =  np.zeros((dimension,2))    
     for idx, line in enumerate(coor_str):    
         entry = line.split()
         for jdx in range(1,3):
             coor[idx][jdx-1] = int(entry[jdx])
             
     denamds =  np.zeros(dimension)         
     for idx, line in enumerate(denamds_str):    
         entry = line.split()
         denamds[idx] = int(entry[1])
             
     dmatrix = np.sqrt(np.sum((coor[:, np.newaxis, :] - coor[np.newaxis, :, :]) ** 2, axis = -1))        
     
        
            
     data = {}
     data['coor'] = coor   
     data['capacity'] = capacity   
     data['dimension'] = dimension   
     data['depot'] = 0
     data['demands'] = denamds
     data['dmatrix'] = dmatrix
     if optimalValue is not None:
         data['optimal'] = int(optimalValue)
     return data