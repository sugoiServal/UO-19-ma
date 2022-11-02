class Apriori():
    def __init__(self, hrz_data, minsup, candidate_fun = 'kk'):
        self.data = hrz_data
        self.minsup = minsup
        self.candidate_fun = candidate_fun
        

    def computeF1(self):
        """
        Compute the initial 1-itemset
        - Input: 
    		- hrz_data: list of N transactions, each transaction is an itemset represented by a int list
    		- minsup: minimum support, given as a proportion （support/#transaction)
    	- Output:
    		- F1: all frequent 1-itemset 
            - F1_tp: all frequent 1-itemset and support count tuple
        
        """
        hrz_data = self.data
        minsup = self.minsup
        C1_support = {}
        num_trans = float(len(hrz_data))
        # scan the database and count support 
        for transaction in hrz_data:
            for item in transaction:
                if not item in C1_support:
                    C1_support[item] = 1
                else:
                    C1_support[item] += 1
        F1 = []
        F1_tp = {}
    
        for (candidate, count) in C1_support.items():
            if (count /num_trans) >= minsup:
                F1.append({candidate})
                F1_tp[frozenset([candidate])] = int(count)			
        return 	F1, F1_tp
    
    def computeCk_kk(self, Fk):
        if len(Fk) == 0:
            return 
        Ck = []
        k = len(Fk[0])
        for i in range(len(Fk)):
            for j in range(i+1, len(Fk)):
                L1 = sorted(list(Fk[i]))[:k-1]
                L2 = sorted(list(Fk[j]))[:k-1]
                if L1 == L2:
                    Ck.append(frozenset(Fk[i].union(Fk[j]).copy()))   
        return Ck

    def computeCk_k1(self, Fk, F1):
        if len(Fk) == 0:
            return 
        Ck = []
        for i in range(len(Fk)):
            for j in range(len(F1)):
                if F1[j].copy().pop() not in Fk[i]:
                    c = Fk[i].union(F1[j])
                    if c not in Ck:
                        Ck.append(frozenset(c))
        return Ck
    

    def solve(self, candidate_fun = 'kk'):
        """
     	Apriori algorithm frequent Itemset generation 
     	- Input: 
    		- hrz_data: list of N transactions, each transaction is an itemset represented by a int list
    		- minsup: minimum support, given as a proportion （support/#transaction)
            - idx_to_items: a mapping from item index to the item's name 
     	- Output:
    		- F: all frequent itemset and support count tuple
            """
        hrz_data = self.data
        minsup = self.minsup
        candidate_fun = self.candidate_fun
        k = 0
        num_trans = float(len(hrz_data))
        (F1, Ftp1) =  self.computeF1()
        F = Ftp1
        Fk = F1
    
        while len(Fk) > 0:
            if candidate_fun == 'kk':
                Ck = self.computeCk_kk(Fk)
            elif candidate_fun == 'k1':
                Ck = self.computeCk_k1(Fk, F1)
            else: raise ValueError("unsupported candidate function")
            k += 1
            print('Running Apriori: the %i-th iteration with %i candidates...' % (k, len(Ck)))
            
            # calculate support count for Ck
            Ck_support = {}
            for transaction in hrz_data:  
                for candidate in Ck:
                    if candidate.issubset(transaction):
                        if not candidate in Ck_support:
                            Ck_support[candidate] = 1
                        else: Ck_support[candidate] += 1
    
            # find frequent k-itemset by support         
            Fk = []   
            for (candidate, count) in Ck_support.items():
                if (count /num_trans) >= minsup:
                    Fk.append({candidate})
                    F[frozenset(sorted(candidate))] = int(count)
        return F
                







    
    