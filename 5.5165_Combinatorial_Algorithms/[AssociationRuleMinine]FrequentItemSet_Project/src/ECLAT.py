import numpy as np
class ECLAT():
    def __init__(self, data, minsup):
        self.minsup = minsup
        self.num_trans = float(len(data))
        self.data = self.computeVerticalDB(data)
        self.support_list = {}

 
        

    def computeVerticalDB(self, hrz_data):
        """ 
        Convert a horizontal database to vertical database, only keep frequent 1-itemset in the database
        
        - Output:
            - vb_F1: frequent 1-itemset in vertical form {()}        
        """      
        

        
        vb_data = {}
        for trans_id, transaction in enumerate(hrz_data):
            for item in transaction:
                if not item in vb_data.keys():
                    vb_data[item] = np.zeros(int(self.num_trans), dtype= int)
                    vb_data[item][trans_id] = 1
                else:
                    vb_data[item][trans_id] = 1   
                
        vb_F1 = {}
        for item in vb_data:
            if vb_data[item].sum()/self.num_trans >= self.minsup:
                vb_F1[item] = vb_data[item]                
        return vb_F1
    

    def solve(self, supportK = None):
        """
        - prefix: k-1 prefix of itemsets
        - supportK: frequent k-itemsets and their tid sets [(itemset (list), bitvector (np array 1*num_trans)), ...]
        """
        # first recursive call
        if supportK is None:
            supportK = []
            # compute supportK from frequent 1-itemsets (supportK, current candidate set)
            for item in self.data:
                supportK.append(([item], self.data[item]))
    
        
        supportK =  sorted(supportK, key=lambda x: int(x[0][-1]))  # visit in alphabetical order of suffix
        #print('Running Eclat in recursive: number of itemsets found: ', len(self.support_list), end='\n')
        
        while len(supportK) > 0:
            itemset, bitvector = supportK.pop(0)
            support = np.sum(bitvector)
            ########### save itemset: {prefix + [itemset]} as frequent itemset
            self.support_list[frozenset(sorted(itemset))] = int(support)
        
        ##################
            suffix = []
            for itemset_sub, bitvector_sub in supportK:
                intersect_bitvector = np.multiply(bitvector, bitvector_sub)
                # only add frequent itemset to the search list
                if np.sum(intersect_bitvector)/self.num_trans >= self.minsup:
                    suffix.append(( sorted(list(set(itemset).union(set(itemset_sub)))), intersect_bitvector )) 
                    
            self.solve(supportK = suffix)
        return self.support_list


