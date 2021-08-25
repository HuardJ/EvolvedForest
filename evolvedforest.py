# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:42:19 2021

@author: Student User
"""
import pandas as pd
from statistics import pvariance,mean
from sklearn.metrics import mean_squared_error,r2_score
from heapq import merge as heap_merge
import copy

def var(X,avg = None):
    
    output = pvariance(X,avg)
        
    return output


def splits_all_cols(dataframe):
    
    splits = dict()
    
    for i in range(len(dataframe.columns)):
        
        splits[i] = list(dataframe.iloc[:,i].drop_duplicates())
        
    return splits


### 


class Node:
    def __init__(self,index = None,child_left = None,child_right = None,
                 feature = None,threshold = None,n_samples = None,
                 impurity = None,val_impurity = None,train_df_slice = None,
                 val_df_slice = None,output_value = None, min_samples_leaf = 1,
                 metric = var,evolution = False,optimal_node = False):
        
        # Node index number
        self.index = index
        
        # Node index to which data less than or equal to the cut 
        # point was passed
        self.child_left = child_left 
        
        # Node index to which data greater than the cut point was was passed
        self.child_right = child_right 
        
        # Feature index the node splits the data on
        self.feature = feature 
        
        # Feature value used as the point for data
        self.threshold = threshold 
        
        # Determine size of traning set either from argument or inference
        if n_samples is None and train_df_slice is not None:
            self.n_samples = len(train_df_slice)
        else: 
            self.n_samples = n_samples
        
        # Training data subset that is passed to this node
        self.train_df_slice = train_df_slice
        
        # validation data subset that is passed to this node
        self.val_df_slice = val_df_slice
        
        # Determine the predicted value this node would return
        # this value can either be passed as an argument or calculated
        # from train_df_slice
        if output_value is None and train_df_slice is not None:
            self.output_value = train_df_slice.iloc[:,-1].mean()
        else:
            self.output_value = output_value
        
        # Determine the error on training data at this node. This 
        # value can either be passed as an argument or calculated 
        # from metric and output_value
        if impurity is None and train_df_slice is not None:
            self.impurity = metric(train_df_slice.iloc[:,-1],
                                   self.output_value)
        else:  
            self.impurity = impurity
        
        # Determine the error on validation data at this node. This 
        # value can either be passed as an argument or calculated 
        # from metric and output_value
        if val_impurity is None and val_df_slice is not None:
            
            self.val_impurity = metric(val_df_slice.iloc[:,-1],
                                       self.output_value)
        else:  
            self.val_impurity = val_impurity
        
        # Tuning parameter - potential cut-points will only be
        # considered if they leave at least min_samples_leaf samples
        # on either side of the split
        self.min_samples_leaf = min_samples_leaf
        
        # Callable function used to determine the error of each node and leaf
        self.metric = metric
        
        # Boolean used to implement evolutionary components
        self.evolution = evolution
        
        # Boolean used by forest class to help select candidate nodes for
        # splitting
        self.optimal_node = optimal_node
        
    
    def splits_per_col(self,col_index):
    
        values = set(self.train_df_slice.iloc[:,col_index])
        
        values = sorted(values)[self.min_samples_leaf-1:-self.min_samples_leaf]
        
        output = pd.Series(values)
        
        return output


    def split_training(self,col_index,split_val):
        
        col_name = self.train_df_slice.columns[col_index]
        
        leq = self.train_df_slice.loc[self.train_df_slice[col_name] <= split_val,:]
        gr = self.train_df_slice.loc[self.train_df_slice[col_name] > split_val,:]
        
        return gr, leq


    def split_validation(self,col_index,split_val):
        
        col_name = self.val_df_slice.columns[col_index]
        
        leq = self.val_df_slice.loc[self.val_df_slice[col_name] <= split_val,:]
        gr = self.val_df_slice.loc[self.val_df_slice[col_name] > split_val,:]
        
        return gr, leq

    # determine the error of the potential split across both
    # partitions of the data
    def overall_error(self,greater,leq):
        N = len(greater) + len(leq)
        
        p_greater = len(greater) / N
        p_leq = len(leq) / N
        
        overall_error = p_greater * self.metric(greater) + p_leq * self.metric(leq)
            
        return overall_error
        
    
    # determine best split for a declared column
    def best_split_col(self, col_index, child_outputs = False, 
                       split_outputs = False):
        
        split_winner = None
        mse_winner = copy.copy(self.impurity) # overall error the best split yields
        
        output = []
        
        splits = self.splits_per_col(col_index)
        for el in splits:
            
            # skip this split in evolution mode if the validation dataframe
            # has no elements in one side of the parition
            if self.evolution:
                
                val_above,val_below = self.split_validation(col_index, el)
                
                if len(val_above) == 0 or len(val_below) == 0:
                    
                    continue
            
            above, below = self.split_training(col_index,el)
                
            mse_challenger = self.overall_error(above.iloc[:,-1],below.iloc[:,-1])
                
            if (split_winner is None) or (mse_challenger < mse_winner):
                
                split_winner = el
                mse_winner = mse_challenger
                
                if child_outputs:
                    gr_node_out = above.iloc[:,-1].mean()
                    leq_node_out = below.iloc[:,-1].mean()
                    
                if split_outputs:
                    df_above = above
                    df_below = below
                    
        if split_winner is None:
            return None
                
        output.append(split_winner)
        output.append(mse_winner)
        
        if child_outputs:
            output.append(gr_node_out)
            output.append(leq_node_out)
            
        if split_outputs:
            output.append(df_above)
            output.append(df_below)
            
        return tuple(output)
    
    
    # Determine whether a candidate node produces worse validation
    # error scores than the current node does, this should reduce
    # overfitting
    def weak_node(self,candidate_score):
        
        return candidate_score > self.val_impurity
    
    
    # Determine what the error for the validation data would be
    # given a specific column, threshold and node output values for
    # child nodes
    def validation_error(self,column_index,threshold,gr_out,
                         leq_out, return_validation_split = False):
        
        above, below = self.split_validation(column_index,
                                             threshold)
        
        #error calculated as sum of residuals squared
        error_above = sum(above.iloc[:,-1].apply(lambda x : (x - gr_out)**2))
        error_below = sum(below.iloc[:,-1].apply(lambda x : (x - leq_out)**2))
        
        output = error_above + error_below
        
        # add validation splits to output if specified
        if return_validation_split:
            output = tuple([output] + [above,below])
        
        return output
        
      
    # Returns the best split values for each feature based on training
    # data
    def all_features_best_splits(self):
    
        column_index = []
        splits = []
        errors = []
        gr_node_out = [] # holds output values of child_right nodes
        leq_node_out = [] # holds output values of child_left nodes
        
        # if self.evolution is true this function also returns the sum of 
        # residual sqares from the validation data, after the split and child 
        # node output values have been applied to it
        if self.evolution:
            validation_sum_residual_squares = []
        
        # running through each column:
        for col_index in range(self.train_df_slice.shape[1]):
            
            # get best split details for this specific column
            temp = self.best_split_col(col_index,
                                       child_outputs = self.evolution)
            
            # check there is a best split for this column
            if temp is None:
                continue # if not then move on to next column
            
            split, mse, gr_out, leq_out = temp # unpack
            
            if self.evolution:
                
                sum_res_sq = self.validation_error(col_index,
                                                    split,
                                                    gr_out,
                                                    leq_out)
                
                # check whether this candidate split is a weak split - 
                # if so, do not add it's details to the lists and skip to the
                # next candidate column
                if self.weak_node(sum_res_sq/len(self.val_df_slice)):
                    continue
                
                # add residual squares score
                validation_sum_residual_squares.append(sum_res_sq)
            
            # add split details
            column_index.append(col_index)
            splits.append(split)
            errors.append(mse)
            gr_node_out.append(gr_out)
            leq_node_out.append(leq_out)
                
            
        ## Wrapping up outputs
        
        if self.evolution:
            
            # Collection of all best candidate splits from each column, with 
            # additional calculated features ordered by their residual 
            # squares score on the validation set
            output = sorted(zip(column_index,
                                splits,
                                errors,
                                gr_node_out,
                                leq_node_out,
                                validation_sum_residual_squares),
                            key = lambda x : x[-1])
                            
            # Add current node index to each candidate split
            output = [(self.index) + el for el  in output]
            
            # Check to see if this node should be split
            # if all further splits start overfitting, then the
            # node is optimal
            if len(output) == 0:
                output = None
                self.optimal_node = True
            
        else: # when self.evolution is not true
            
            # select best performing candidate split from training errors
            output = min(zip(column_index,
                             splits,
                             errors,
                             gr_node_out,
                             leq_node_out),
                         key = lambda x : x[2])
            
            output = (self.index) + output
            
        return output
        

class Tree:
    def __init__(self,train_df_slice,val_df_slice,nodes = None,
                 changed = None,min_samples_split = 2,
                 evolution = False,min_samples_leaf = 1):
        
        # List of nodes in tree, if nodes argument passed is None
        # then create a list with just the base node
        if nodes is None:
            self.nodes = [Node(index = 0,
                               train_df_slice = train_df_slice.iloc[:,:],
                               val_df_slice = val_df_slice.iloc[:,:],
                               min_samples_leaf = min_samples_leaf,
                               evolution = evolution)]
            
        #Store original size of train_df_slice and val_df_slice
        self.train_N = len(train_df_slice)
        self.val_N  = len(val_df_slice)
        
        # Boolean to determine whether any nodes have split since
        # previous generation. If not then the tree is considered
        # optimal
        self.changed = changed
        
        # Tuning parameter - number of samples a node must 
        # contain in order to consider splitting
        self.min_samples_split = min_samples_split
        
        # Tuning parameter - potential cut-points will only be
        # considered if they leave at least min_samples_leaf samples
        # on either side of the split
        self.min_samples_leaf = min_samples_leaf
        
        # Boolean - alters algorithm accordingly
        self.evolution = evolution
        
        
    # Returns list of nodes with only those able to be split
    def nodes_to_split(self):
        
        output = [el for el in self.nodes
                  if el.feature is None
                  and len(el.train_df_slice) > self.min_samples_split
                  and el.optimal_node is False]
        
        return output
        
    # Evolution method - Returns potential node-split combinations from each
    # node in tree ordered by validation score
    def next_gen_seeds(self,n_estimators):
        
        current_val_score = self.validation_score(False)
        
        splittable = self.nodes_to_split()
        
        # place holder for best performing seed
        base_seed = []
        
        all_node_features = []
        
        # run through each splittable node
        for node in splittable:
            
            temp = node.all_features_best_splits()
            
            # check node is not optimal
            if temp is not None:
                
                # add best performing split to base_seed and remove from temp
                base_seed.append(temp.pop(0))
                
                # add alternative splits to all_node_features - ordered by
                # validation scores
                all_node_features = heap_merge(all_node_features,temp,
                                               key = lambda x : x[-1])
        
        # if all nodes are optimal, then set changed to false and return None
        if len(base_seed) == 0:
            self.changed = False
            return None
        
        ### output is an ordered list of best performing seeds and the tree
        ### validation score if seed is grown
        
        # list of differences in validation score between current and new node 
        # splits
        new_val_score = [em.val_impurity * len(em.val_df_slice) - el[-1] 
                         for el in base_seed for em in self.nodes
                         if el[0] == em.index]
        
        # current validation score minus sum of differences
        new_val_score = current_val_score - sum(new_val_score)
        
        output = [(base_seed,new_val_score)]
        
        # condensed list from all_node_features of node-feature split index 
        # and the difference in validation scores this node-feature split and 
        # this nodes best split, which is found in best_seed.
        idx_val = [((i),all_node_features[i][-1] - j)
                   for i in range(len(all_node_features)) 
                   for j in [el[-1] for el in base_seed 
                             if el[0] == all_node_features[i][0]]]
        
        # order index_val by the amount of additional validation error from 
        # base seed each node-feature split results in
        idx_val = sorted(idx_val,key = lambda x : x[-1])
        
        # only need at most n_estimators worth of seeds and we already have 
        # base_seed
        if len(idx_val) > n_estimators - 1:
            idx_val = idx_val[:n_estimators]
            
        # add seeds in order of best performing overall node-feature 
        # combinations
        i = 0 # number of elements in idx_val already considered 
        while len(output) < n_estimators:
            
            # build next_seed off of base seed, replacing node-feature splits
            # in base seeds with next best splits determined by idx_val
            next_seed = base_seed
            
            # cycle through all_node_features indexes stored in idx_val at 
            # point i and retreive corresponding node feature splits from 
            # all_node_features
            for idx in idx_val[i][0]:
                
                replacement_node = all_node_features[idx]
                
                # find and replace corresponding node in next_seed
                j = 0
                while j < len(next_seed):
                    if next_seed[j][0] == replacement_node[0]:
                        next_seed[j] = replacement_node
                        j = len(next_seed) # stop looping
                    else:
                        j += 1 # increment
                        
            # list of differences in validation score between current and new
            # node splits
            new_val_score = [em.val_impurity * len(em.val_df_slice) - el[-1] 
                             for el in next_seed for em in self.nodes
                             if el[0] == em.index]
            
            # current validation score minus sum of differences
            new_val_score = current_val_score - sum(new_val_score)
                        
            # insert next_seed into output
            output.append((next_seed,new_val_score))
                        
            # update idx_val with combinations of previous features
            new_idx_vals = []
            j = i - 1 # previous idx_val considered
            while j >= 0:
                # check no overlap between index components
                if len(set(idx_val[i][0]) & set(idx_val[j][0])) == 0:
                    new_combination = (idx_val[i][0] + idx_val[j][0],
                                       idx_val[i][1] + idx_val[j][1])
                    
                    new_idx_vals.append(new_combination)
                    
                j -= 1 # increment
                    
            # sort new_idx_vals and add to idx_val
            idx_val = heap_merge(idx_val,
                                 sorted(new_idx_vals,key = lambda x : x [-1]),
                                 key = lambda x : x[-1])
            
            # condense idx_val
            if len(idx_val) > self.n_estimators:
                idx_val = idx_val[:self.n_estimators]
            
            # increment i
            i += 1
            # end of while loop
            
        return output
    
    
    # Return node by nodes index attribute
    def get_node(self, idx, from_splittable = False):
        
        output = None
        
        if from_splittable:
            node_list = self.nodes_to_split()
        else:
            node_list = self.nodes
        
        for node in node_list:
            if node.index == idx:
                output = node
                break
            
        return output
    
    
    # Evolution method - builds the tree according to specified
    # node and feature to split on.
    def grow_specific(self,node_feature_pairs):
        
        # ensure node_feature_pairs argument was passed as list type
        # containing tuples or lists of size 2
        not_list = type(node_feature_pairs) is not list
        
        check_elements = all([type(el) in [list,tuple] for el 
                              in node_feature_pairs])
        
        check_len = all(len(el) == 2 for el in node_feature_pairs)
        
        if not_list and not check_elements and not check_len:
            print('not_list:',not_list)
            print('check_elements:',check_elements)
            print('check_len:',check_len)
            raise ValueError('Invalid node_feature_pairs argument')
        
        # Specify that the tree has not changed yet, this will be
        # updated if candidate splits are deemed good
        self.changed = False
        
        # cycle through features and splittable nodes
        for node_feature in node_feature_pairs:
            
            # select feature and node
            node = self.get_node(node_feature[0],True)
            feature = node_feature[1]
            
            # Check node exists i.e. was a splittable node. Otherwise, skip
            # to next node_feature
            if node is None:
                continue
            
            # collect up arguments to be passed to node and 
            # node's children
            
            # find best split for specified feature
            temp = node.best_split_col(feature,
                                       child_outputs = True,
                                       split_outputs = True)
            
            # unpack temp
            split, error, mean_gr, mean_leq, train_gr, train_leq = temp
            
            # split validation data according to best split
            temp = node.validation_error(column_index = feature,
                                         threshold = split,
                                         gr_out = mean_gr,
                                         leq_out = mean_leq,
                                         return_validation_split = True)
            
            validation_error,validation_gr,validation_leq = temp # unpack temp
            
            # check split is not weak - if split is weak, then do 
            # not split node
            if node.weak_node(validation_error/len(node.val_df_slice)):
                continue
            
            # If the split is not weak then the tree is changing
            self.changed = True
            
            # updated node's attributes and insert child nodes to
            # nodes list (Tree attribute)
            
            node.feature = feature
            node.threshold = split
            
            node.child_left = len(self.nodes)
            self.nodes.append(Node(index = len(self.nodes),
                                   train_df_slice = train_leq,
                                   val_df_slice = validation_leq,
                                   output_value = mean_leq,
                                   evolution = True))
            
            node.child_right = len(self.nodes)
            self.nodes.append(Node(index = len(self.nodes),
                                   train_df_slice = train_gr,
                                   val_df_slice = validation_gr,
                                   output_value = mean_gr,
                                   evolution = True))
            
            node.train_df_slice = None
            node.val_df_slice = None
            node.output_value = None
            
        return None
    
    
    def grow_seed(self,seed):
        
        for node_split_features in seed:
            
            node_to_split = self.get_node(node_split_features[0])
            
            node_to_split.feature = node_split_features[1]
            node_to_split.threshold = node_split_features[2]
            
            # split training dataframe slice
            temp = node_to_split.split_training(node_split_features[1],
                                                node_split_features[2])
            
            train_gr,train_leq = temp # Unpack
            
            # split validation dataframe slice
            temp = node_to_split.split_validation(node_split_features[1],
                                                  node_split_features[2])
            
            validation_gr,validation_leq = temp # Unpack
            
            # empty slice's and ouput value
            node_to_split.train_df_slice = None
            node_to_split.val_df_slice = None
            node_to_split.output_value = None
            
            # Add new nodes
            node_to_split.child_left = len(self.nodes)
            self.nodes.append(Node(index = len(self.nodes),
                                   train_df_slice = train_leq,
                                   val_df_slice = validation_leq,
                                   output_value = node_split_features[4],
                                   evolution = True))
            
            node_to_split.child_right = len(self.nodes)
            self.nodes.append(Node(index = len(self.nodes),
                                   train_df_slice = train_gr,
                                   val_df_slice = validation_gr,
                                   output_value = node_split_features[5],
                                   evolution = True))
            
            self.changed = True
        
        return None
    
    
    def validation_score(self,scaled = True):
        
        ### Validation_score currently only returns sum of squared errors or
        ### mean squared error
        
        # Find leaf nodes
        leaf_nodes = [node for node in self.nodes if node.feature is None]
        
        # find the sum of squared residuals from each node
        output = sum([len(node.val_df_slice) * node.val_impurity for node in leaf_nodes])
        
        # mean square error
        if scaled:
            output = output/self.val_N
        
        return output
    
    
    def copy(self):
        
        new_tree = copy.copy(self)
        new_tree.nodes = [copy.copy(el) for el in new_tree.nodes]
        
        return new_tree
        
    
    def predict(self,X):
        
        records = X.shape[0]
        result = []
        for i in range(records):
            
            node_id = 0
            new_node_id = 0
            while new_node_id is not None:
                
                node_id = new_node_id
                if X.iloc[i,self.nodes[node_id].feature] <= self.nodes[node_id].threshold:
                    new_node_id = self.nodes[node_id].child_left
                else: new_node_id = self.nodes[node_id].child_right
                
            result.append(self.nodes[node_id].output_value)
        
        return pd.Series(result)
                    
                    
class EvolvedForest:
    
    def __init__(self,n_estimators = 100,initial_depth = 2,
                 max_depth = None,min_samples_split = 2,
                 min_samples_leaf = 1):
        
        # number of trees
        self.n_estimators = n_estimators
        
        # Overide how deep the trees must go in their first generation
        self.initial_depth = initial_depth
        
        # Determine depth of trees - if none then keep splitting
        # until the data in each leaf fully coincides or until
        # other tuning parameters constrain further splits
        self.max_depth = max_depth
        
        # Tuning parameter - number of samples a node must 
        # contain in order to consider splitting
        self.min_samples_split = min_samples_split
        
        # Tuning parameter - potential cut-points will only be
        # considered if they leave at least min_samples_leaf samples
        # on either side of the split
        self.min_samples_leaf = min_samples_leaf
        
        # Container for trees - initialize with a single tree
        self.forest = []
        
        
    # How many node_feature combinations can be initially tried 
    # given the number of estimators specified? 
    # The following function helps determine this by calculating
    # the depth of trees in their first generation
    def enumerate_features(self,train_data):
        
        n_features = train_data.shape[1]
        depth = 0
        
        # Each node has n_feature possibilities. Each depth level 
        # has 2 times the number of nodes of the previous level
        while (n_features**(depth + 1)) * (2**(depth)) < self.n_estimators:
            depth += 1
            
        return depth
    
    
    # returns lists of node_pair combinations to grow
    def feature_combinations(self,n_features,depth):
        
        # determine the node indices
        nodes = range(2**depth - 1, 2**(depth + 1) - 1)
        
        # feature index list
        features = range(n_features)
        
        # feature combinations for first node in nodes
        node_features = [[(nodes[0],el)] for el in features]
        
        # assign feature pairs for each node and node combination
        current_node = 1
        while current_node < len(nodes):
            
            node_features = [el1 + [(nodes[current_node],el2)] 
                             for el2 in features for el1 in node_features]
            current_node += 1
            
        return node_features
    
            
    def train(self,train_X,train_y,val_X,val_y,verbose = False):
        
        # feature space dimensions and total training data size
        training_N, training_features_n = train_X.shape
        
        ### PHASE 1: Build initial set of trees
        if verbose:
            print('Phase 1 started')
            
        # Calculate inititial trees' depth
        depth = self.enumerate_features(train_X)
        
        # determine whether calculated initial tree depth is too
        # small and should be overriden by pre-set tree depth
        if depth < self.initial_depth:
            depth = self.initial_depth # overide
            
        if verbose:
            print('\tPhase 1 tree depth: ', depth)
        
        # combine training data into a whole entity
        train_X['y'] = train_y
        
        # combine validation data into a whole entity
        val_X['y'] = val_y
        
        if verbose:
            print('Growing first generation trees...')
            
        # Create a single tree as template
        self.forest.append(Tree(train_X.iloc[:,:],
                                val_X.iloc[:,:],
                                min_samples_split = self.min_samples_split,
                                min_samples_leaf = self.min_samples_leaf,
                                evolution = True))
        
        # Create and Grow Tree for each feature set
        for level in range(depth + 1):
            
            if verbose:
                print('\tCreating ',str(level),' depth features')
                
            new_forest = []
            
            # get node-feature pairs for this depth level
            node_features = self.feature_combinations(training_features_n
                                                      ,level)
            
            # Cycle through currently existing forest
            for tree in self.forest:
                
                # Cycle throught node-feature permutations
                for n_f in node_features:
                    
                    # take deep copy of tree to leave the original unaltered and
                    # able to be further referenced
                    temp_tree = tree.copy()
                    
                    temp_tree.grow_specific(n_f)
                    
                    # Add grown temp_tree to the new_forest
                    new_forest.append(temp_tree)
                    
            # replace current forest with new_forest
            self.forest = new_forest
            
        
        # Since initial feature set is larger than n_estimators, select the 
        # top 'n_estimators' performing trees based on validation_score
        self.forest = sorted(self.forest,key = lambda x : x.validation_score())
        self.forest = self.forest[:self.n_estimators]
        
        if verbose:
            print('Phase 2...')
            
        ### PHASE 2: Build while loop to keep splitting trees until none can 
        ### be split anymore
        while any([tree.changed for tree in self.forest]):
            
            if verbose:
                print('\tGenerating seeds...')
                
            ### PHASE 3: Generate next set of viable splits for each tree and
            ### select best performing splits
            
            seed_bag = [] # holds seeds from any tree
            for i in range(len(self.forest)):
                
                # generate seeds from trees not known to be unchangeable
                if self.forest[i].changed:
                    seeds = self.forest[i].next_gen_seeds()
                    
                    # if seeds is not None then add seeds to seed_bag
                    if seeds is not None:
                        
                        # make note of parent tree
                        seeds = [(i) + el for el in seeds]
                        
                        seed_bag = heap_merge(seed_bag,seeds,
                                              key = lambda x : x[-1])
                        
                        # seed_bag should only hold at most n_estimators
                        if len(seed_bag) > self.n_estimators:
                            seed_bag = seed_bag[:self.n_estimators]
            
            if verbose:
                print('\tDetermining Successors...')
                
            ### PHASE 4: compare best performing splits with currently optimal
            ### trees
            
            # get seedless trees validation scores
            seedless_trees = [i for i in range(len(self.forest)) 
                              if self.forest[i].changed == False]
            
            seedless_trees = [(el,self.forest[el].validation_score(False))
                              for el in seedless_trees]
            
            # determine successor trees and seeds
            successors = heap_merge(seedless_trees,
                                    seed_bag,
                                    key = lambda x : x[-1])
            successors = successors[:self.n_estimators]
            
            if verbose:
                print('\tGrowing Successors...')
                
            ### PHASE 5: Grow seeds and create new forest
            
            new_forest = []
            
            # loop through successors to select trees from index values
            for successor in successors:
                
                # check whether successor is optimal tree or seed
                if len(successor) == 3: # seed
                
                    # if seed then make a deepcopy of associated tree and split 
                    # nodes in seed
                    new_tree = self.forest[successor[0]].copy()
                    new_tree.grow_seed(successor[1])
                    
                    # add tree to new_forest
                    new_forest.append(new_tree)
                else:
                    # otherwise add optimal tree to new forest
                    new_forest.append(self.forest[successor[0]])
                    
            self.forest = new_forest
                    
        return self
        
        
    def predict(self,X):
            
        predicted_values = 0
        for tree in self.forest:
            temp = tree.predict(X)
            predicted_values += temp
            
        predicted_values = predicted_values/len(self.forest)
        
        return predicted_values
    
    
    def prediction_score(self,X,y):
        
        predicted_values = self.predict(X)
        output = r2_score(y, predicted_values)
        
        return output
        