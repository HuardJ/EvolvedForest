# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:42:19 2021

@author: Student User
"""
import pandas as pd
from statistics import variance,mean
from sklearn.metrics import mean_squared_error

def var(X):
    if len(X) > 1:
        output = variance(X)
    else:
        output = 0
        
    return output


def splits_all_cols(dataframe):
    
    splits = dict()
    
    for i in range(len(dataframe.columns)):
        
        splits[i] = list(dataframe.iloc[:,i].drop_duplicates())
        
    return splits


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
        if impurity is None and val_df_slice is not None:
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
    
        column_name = self.train_df_slice.columns[col_index]
        
        temp = self.train_df_slice.sort_values(column_name)
        temp.reset_index(drop = True,inplace = True)
        
        splits = temp.iloc[self.min_samples_leaf-1:-self.min_samples_leaf
                           ,col_index]
        
        return list(splits.unique())


    def split_training(self,col_index,split_val):
        
        col_name = self.train_df_slice.columns[col_index]
        
        leq = self.train_df_slice.loc[self.train_df_slice[col_name] <= split_val,:]
        gr = self.train_df_slice.loc[self.train_df_slice[col_name] > split_val,:]
        
        return gr, leq


    def split_validation(self,col_index,split_val):
        
        col_name = self.val_df_slice.columns[col_index]
        
        leq = self.train_df_slice.loc[self.val_df_slice[col_name] <= split_val,:]
        gr = self.train_df_slice.loc[self.val_df_slice[col_name] > split_val,:]
        
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
        mse_winner = None # overall error the best split yields
        
        output = []
        for el in self.splits_per_col(col_index):
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
        
        # if self.evolution is true this function returns the following
        # additional values: the sum of residual sqares from the 
        # validation data when the split and child node output values
        # have been applied to it
        if self.evolution:
            validation_sum_residual_squares = []
        
        # running through each column:
        for col_index in range(self.train_df_slice.shape[1]):
            
            column_index.append(col_index)
            
            temp = self.best_split_col(col_index,
                                       child_outputs = self.evolution)
            
            split, mse, gr_out, leq_out = temp
            
            splits.append(split)
            errors.append(mse)
            gr_node_out.append(gr_out)
            leq_node_out.append(leq_out)
            
            if self.evolution:
                
                sum_res_sq = self.validation_error(col_index,
                                                   split,
                                                   gr_out,
                                                   leq_out)
                
                validation_sum_residual_squares.append(sum_res_sq)
                
        if self.evolution:
            
            # Collection of all best candidate splits from each
            # column, with additional calculated features
            # ordered by their residual squares score on the validation set
            output = sorted(list(zip(column_index,
                                     splits,
                                     errors,
                                     gr_node_out,
                                     leq_node_out,
                                     validation_sum_residual_squares)),
                            key = lambda x : x[-1])
                            
            # Add current node index to each candidate split
            output = [(self.index) + el for el  in output]
            
            # remove nodes that have a worse validation score this current node
            # this should reduce overfitting
            output = [el for el in output if not self.weak_node(el[-1])]
            
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
        
    # Evolution method - Returns potential splits from each
    # node in tree
    def next_gen_seeds(self):
        
        splittable = self.nodes_to_split()
        
        output = []
        for node in splittable:
            
            temp = node.all_features_best_splits()
            
            if temp is not None:
                
                output.append(temp)
                
        return output
    
    
    # Evolution method - builds the tree according to specified
    # features
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
            node = node_feature[0]
            feature = node_feature[1]
            
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
            if node.weak_node(validation_error):
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
                                   output_value = mean_leq))
            
            node.child_right = len(self.nodes)
            self.nodes.append(Node(index = len(self.nodes),
                                   train_df_slice = train_gr,
                                   val_df_slice = validation_gr,
                                   output_value = mean_gr))
            
            node.train_df_slice = None
            node.val_df_slice = None
            node.output_value = None
            
        return None
                    
    
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
    
    def __init__(self,n_estimators = 100):
        
        self.n_estimators = n_estimators
        self.trees = []
        # for i in range(n_estimators):
        #     self.trees.append(Tree())
        
        
    # How many node_feature combinations can be initially tried 
    # given the number of estimators specified? 
    # The following function helps determine this by calculating
    # the depth of trees in their first generation
    def enumerate_features(self,train_data):
        
        n_features = train_data.shape[1]
        depth = 1
        while (n_features**depth) * (2**(depth - 1)) < n_estimators:
            depth += 1
            
        return depth
    
    
    def feature_combinations(train_data,depth):
        
        original_list = list(range(train_data.shape[1]))
        original_list = [[el] for el in original_list]
        
        feat_list = original_list.copy()
        
        iterate = 2
        while iterate <= depth:
            feat_list = [el1 + el2 for el2 in original_list for el1 in feat_list]
            iterate += 1
            
        return feat_list
    
            
    def train(self,train_X,train_y,val_X,val_y):
        
        training_features_n = train_X.shape[1]
        #print('Number of training features: ',training_features_n)
        
        survival_number = int(self.n_estimators/training_features_n)
        #print('Number of trees to survive: ', survival_number)
        
        ### PHASE 1: Build initial set of trees
        depth = enumerate_features(train_X,self.n_estimators)
        #print('Phase 1 tree depth: ', depth)
        
        features = feature_combinations(train_X, depth)
        
        train_X['y'] = train_y
        
        for feat in features:
            #print('Creating tree for ',feat,' features')
            self.trees.append(Tree())
            self.trees[-1].grow_specific(train_X, feat)
            
            
        
        ### PHASE 2: Build while loop to keep splitting trees until none can be split anymore
        while any([tree.changed for tree in self.trees]):
            
            ### PHASE 3: Evaluate forest against cross-validation
            score = dict()
            for i in range(len(self.trees)):
                
                predicted_validation = self.trees[i].predict(val_X)
                validation_score = mean_squared_error(val_y,predicted_validation)
                
                score[i] = validation_score
                
            score = sorted(score.items(),key = lambda x : x[1])
            
            
            # PHASE 4: Select best performing trees to survive and replace old forest
            surviving_trees_idx = [el[0] for el in score[:survival_number + 1]]
            
            next_gen_trees = []
            for el in surviving_trees_idx:
                
                next_gen_trees += [self.trees[el]]*training_features_n
                
            self.trees = next_gen_trees
            
            # Phase 5: Add new split to each tree
            for i in range(len(self.trees)):
                self.trees[i].grow_specific(train_X, i%training_features_n)
        
        
    def predict(self,X):
            
        predicted_values = 0
        for tree in self.trees:
            temp = tree.predict(X)
            predicted_values += temp
            
        predicted_values = predicted_values/len(self.trees)
        
        return predicted_values