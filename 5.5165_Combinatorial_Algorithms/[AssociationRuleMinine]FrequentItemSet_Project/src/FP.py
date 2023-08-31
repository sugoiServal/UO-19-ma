class Node:
    def __init__(self, item, count=0, parent=None, link=None):
        """
        Nodes in FP-tree
        - Input:
            - item: positive int, -1 refers to empty set(null)
            - count: number of times the item occurs in a path 
        """
        self.item=item
        self.count=count
        self.parent=parent
        self.link=link
        self.children={}

              
   
class FPtree:
    def __init__(self, hrz_data, minsup=0.1, num_trans = None):
        """
        a FP-tree or conditional FP-tree
        - Input:
            - hrz_data: horizontal database 
            - num_trans: number of transactions in the initial database
            - minsup: minimum support, given as a proportion （support/#transaction)
        """
        

        self.data = hrz_data
        self.minsup = minsup
        if num_trans is None:
            self.num_trans = float(len(hrz_data))
        else: self.num_trans = num_trans
        
        # null root
        self.root = Node(item=-1, count=1)
               
        ################## link list table ######################
        # link table containing initial node in the link-list of nodes-of-same-item
        self.link_table=[]
        
        # dictionaly containing frequent 1-itemset and the support count        
        self.F1={}
                
        # dictionary with item and it's rank in support count 
        self.item_rank={}

        ############## construct the initial FP-tree, stored in self.root ##################
        self.constFPtree()

    def constFPtree(self):
        """
        Initialize all tables; Construct the inital FP-tree; Construct node link list;
        through two scan of database
        
        """
        
        ## initialize frequent 1-itemset F1 in the first scan
        
        # get support count for 1-itemset
        data = self.data      
        for transaction in data:
            for item in transaction:
                if not item in self.F1:
                    self.F1[item] = 1                   
                else:
                    self.F1[item] += 1   
                    
        # prune non-frequent itemset support_count_1            
        all_items = list(self.F1.keys())
        
        for item in all_items:
            if(self.F1[item]/ self.num_trans < self.minsup):
                del self.F1[item]
                
        # reorganize and sort as tuple in decreasing frequency order 
        itemortdic = sorted(self.F1.items(), key=lambda x: (-x[1],x[0])) 
        
        ## initialize item_rank and link_table
        rank = 0
        for (item, support_count) in itemortdic:
            self.item_rank[item] = rank
            rank += 1
            item_info = {'item':item, 'support':support_count, 'linknode': None}
            self.link_table.append(item_info)
            
        ## construct FP-tree in the second scan
    
        for tranaction in data:
            fq_items = []
            for item in tranaction:
                # only keep frequent item from the transaction 
                if item in self.F1.keys():
                    fq_items.append(item)         
                    
            if len(fq_items)>0:
                # order the remaining items in transaction in increasing item frq rank order
                sorted_tranaction = sorted(fq_items, key = lambda k: self.item_rank[k])

                # build the tree with current [sorted_tranaction] 
                cNode = self.root
                for cur_item in sorted_tranaction:         
                    # when available path exist in the FP-tree, use pre-existing path
                    if cur_item in cNode.children.keys():
                        cNode.children[cur_item].count += 1
                        cNode = cNode.children[cur_item]
                        
                    # when not path available, create new path by branching
                    else:
                        # create node
                        cNode.children[cur_item] = Node(item=cur_item, count=1, parent=cNode, link=None)
                        cNode = cNode.children[cur_item]
                        # link this node to link_table
                        for item_info in self.link_table:
                            # find the link list correspond to cNode.item
                            if item_info["item"] == cNode.item:                              
                                # if it is the first node in link list
                                if item_info["linknode"] is None:
                                    item_info["linknode"] = cNode
                                    
                                # else find the last node of the item' link list and add the current node 
                                else:
                                    iter_node = item_info["linknode"]
                                    while(iter_node.link is not None):
                                        iter_node = iter_node.link
                                    iter_node.link = cNode
 
    def condTreeTran(self, N):
        """
        construct conditinal FP-tree   
        
        """
        if N.parent is None:
            return None
        
        condtreeline =[]
        # starting from the leaf node (N) treveras each path in the current FP-tree and add item till hit root
        # jump between paths by N->link
        while N is not None:
            line=[]
            PN = N.parent
            while PN.parent is not None:
                line.append(PN.item)
                PN=PN.parent
            # reverse order and store the transaction
            line = line[::-1]
            for i in range(N.count):
                condtreeline.append(line)   
            # to next leaf
            N=N.link
        return condtreeline
    
    

    
    def solve(self, parent = None):
        """
        
        Generate frequent itemset from FP-tree by recursively generate FP-tree 
        
        """
        # empty FP-tree
        if len(list(self.root.children.keys())) == 0:
            return None
        
        result = {}       
        # visit suffix by decreasing frequency order of item, i.e. visit the reversed link_table
        revtable = self.link_table[::-1]
        
        for n in revtable:
            # determine frequent itemset in current tree
            fqset = [set(),0]
            # 1-itemset
            if(parent == None):      
                fqset[0] = {n['item'],}
            # current item + suffix
            else:
                fqset[0] = {n['item']}.union(parent[0])
            fqset[1] = n['support']
            result[frozenset(sorted(list(fqset[0])))] = fqset[1]
            
            # generate conditaion tree
            cond_tran = self.condTreeTran(n['linknode'])
                        
            # recursively build the conditinal FP-tree and generate freq itemset
            contree= FPtree(cond_tran, self.minsup, self.num_trans)
            sub_res = contree.solve(parent = fqset)
            if sub_res is not None:
                for itemset in sub_res:
                    if itemset not in result.keys():
                        result[itemset] = sub_res[itemset]
                 
        return result



