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


def splits_per_col(dataframe,col_index):
    
    splits = list(dataframe.iloc[:,col_index].drop_duplicates())
    
    #Remove Largest value from list, unecessary for split:
    splits.remove(max(splits))
    
    return splits


def splits_all_cols(dataframe):
    
    splits = dict()
    
    n_cols = len(list(dataframe))
    
    for i in range(n_cols):
        splits[str(i)] = list(dataframe.iloc[:,i].drop_duplicates())
        
    return splits


def split_col(dataframe,col_index,split_val):
    col_name = list(dataframe)[col_index]
    
    leq = dataframe.loc[dataframe[col_name] <= split_val, col_name]
    gr = dataframe.loc[dataframe[col_name] > split_val, col_name]
    
    return gr,leq


def split_df(dataframe,col_index,split_val):
    col_name = list(dataframe)[col_index]
    
    leq = dataframe.loc[dataframe[col_name] <= split_val,:]
    gr = dataframe.loc[dataframe[col_name] > split_val,:]
    
    return gr, leq


def calculate_overall_metric(greater,leq,metric = var):
    N = len(greater) + len(leq)
    
    p_greater = len(greater) / N
    p_leq = len(leq) / N
    
    overall_metric = p_greater * metric(greater) + p_leq * metric(leq)
    
    return overall_metric


def best_split_col(dataframe, col_index, return_error = False):
    potential_splits = splits_per_col(dataframe,col_index)
    
    split_winner = None
    mse_winner = None
    for el in potential_splits:
        above, below = split_df(dataframe,col_index,el)
        
        mse_challenger = calculate_overall_metric(above.iloc[:,-1],below.iloc[:,-1])
        
        if (split_winner is None) or (mse_challenger < mse_winner):
            split_winner = el
            mse_winner = mse_challenger
            
        if return_error:
            return split_winner,mse_winner
        else: return split_winner


def best_split_all_features(dataframe):
    
    col_winner = None
    split_winner = None
    mse_winner = None
    for col_index in range(dataframe.shape[1]):
        
        split_challenger,mse_challenger = best_split_col(dataframe,col_index,return_error = True)
        
        if (split_winner is None) or (mse_challenger < mse_winner):
            col_winner = col_index
            split_winner = split_challenger
            mse_winner = mse_challenger
            
    return col_winner,split_winner


def enumerate_features(train_data,n_estimators):
    
    n_features = train_data.shape[1]
    #round_one_trees = 0
    depth = 1
    while n_features**depth < n_estimators:
        
        #round_one_trees = n_features**depth
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


class Node:
    def __init__(self,idx = None,left = None,right = None,feat = None,
                 thresh = None,n_samples = None,imp = None,l_r_slice = None,
                 out_val = None):
        
        self.index = idx
        self.child_left = left
        self.child_right = right
        self.feature = feat
        self.threshold = thresh
        self.n_node_samples = n_samples
        self.impurity = imp
        self.data_slices = l_r_slice 
        self.output_value = out_val
        

class Tree:
    def __init__(self,nods = None,change = None):
        self.nodes = nods
        self.changed = change
    
    
    def grow_specific(self,data,features,min_samples_split = 2,min_samples_leaf = 1):
        if type(features) is not list:
            features = [features]
        
        self.changed = False
        for feature in features:
            #print('Build node for ',feature)
            if self.nodes is None:
                split,mse = best_split_col(data,feature,True)
                above,below = split_df(data,feature,split)
                
                self.nodes = [Node(0,feat = feature,thresh = split,n_samples = len(data),
                                   imp = mse,l_r_slice = (below,above),
                                   out_val = mean(data.iloc[:,-1]))]
                
                self.changed = True
                
            else:
                nodes_to_split = [el for el in self.nodes if el.child_left is None and el.n_node_samples >= min_samples_split]
                
                for node in nodes_to_split:
                    
                    if min(node.data_slices[0].iloc[:,feature].nunique(),
                           node.data_slices[1].iloc[:,feature].nunique()) > min_samples_leaf:
                        
                        self.changed = True
                        
                        node.child_left = len(self.nodes)
                        split,mse = best_split_col(node.data_slices[0],feature,True)
                        above,below = split_df(node.data_slices[0],feature,split)
                        
                        self.nodes.append(Node(len(self.nodes),
                                               feat = feature,
                                               thresh = split,
                                               n_samples = len(node.data_slices[0]), 
                                               imp = mse,
                                               l_r_slice = (below,above),
                                               out_val = mean(node.data_slices[0].iloc[:,-1])))
                        
                        
                        node.child_right = len(self.nodes)
                        split,mse = best_split_col(node.data_slices[1],feature,True)
                        above,below = split_df(node.data_slices[1],feature,split)
                        
                        self.nodes.append(Node(len(self.nodes),
                                               feat = feature,
                                               thresh = split,
                                               n_samples = len(node.data_slices[1]), 
                                               imp = mse,
                                               l_r_slice = (below,above),
                                               out_val = mean(node.data_slices[1].iloc[:,-1])))
                    
                        node.data_slices = None
                        node.output_value = None
                    
    
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