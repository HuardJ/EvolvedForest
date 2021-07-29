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


def splits_per_col(dataframe,col_index,min_samples_leaf = 1):
    
    column_name = dataframe.columns[col_index]
    
    splits = dataframe.sort_values(column_name).reset_index(drop = True).iloc[min_samples_leaf-1:-min_samples_leaf,col_index].drop_duplicates()    
    
    return list(splits)


def splits_all_cols(dataframe):
    
    splits = dict()
    
    for i in range(len(dataframe.columns)):
        
        splits[i] = list(dataframe.iloc[:,i].drop_duplicates())
        
    return splits


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
    def __init__(self,index = None,child_left = None,child_right = None,feature = None,
                 threshold = None,n_samples = None,impurity = None,train_df_slice = None,
                 val_df_slice = None,output_value = None,metric = var, evolution = False):
        
        self.index = index
        self.child_left = child_left
        self.child_right = child_right
        self.feature = feature
        self.threshold = threshold
        
        if n_samples is None and train_df_slice is not None:
            self.n_samples = len(train_df_slice)
        else: 
            self.n_samples = n_samples
        
        self.impurity = impurity
        self.train_df_slice = train_df_slice
        self.val_df_slice = val_df_slice
        self.output_value = output_value
        self.metric = metric
        self.evolution = evolution


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


    def calculate_overall_metric(self,greater,leq):
        N = len(greater) + len(leq)
        
        p_greater = len(greater) / N
        p_leq = len(leq) / N
        
        overall_metric = p_greater * self.metric(greater) + p_leq * self.metric(leq)
        gr_mean = mean(greater)
        leq_mean = mean(leq)
            
        return overall_metric,gr_mean,leq_mean
        
        
    def best_split_col(self, col_index, min_samples_leaf = 1):
        potential_splits = splits_per_col(self.train_df_slice,col_index,min_samples_leaf = min_samples_leaf)
        
        split_winner = None
        mse_winner = None
        gr_node_val = None
        leq_node_val = None
        for el in potential_splits:
            above, below = self.split_training(col_index,el)
            
            mse_challenger,gr_mean,leq_mean = self.calculate_overall_metric(above.iloc[:,-1],below.iloc[:,-1],True)
                
            if (split_winner is None) or (mse_challenger < mse_winner):
                
                split_winner = el
                mse_winner = mse_challenger
                gr_node_val = gr_mean
                leq_node_val = leq_mean
            
        return split_winner,mse_winner,gr_node_val,leq_node_val
        
        
    def all_features_best_splits(self,min_samples_leaf = 1):
    
        splits = []
        errors = []
        gr_node_out = []
        leq_node_out = []
        
        if self.evolution:
            validation_sum_residual_squares = []
        
        for col_index in range(self.train_df_slice.shape[1]):
            
            split, mse, gr_out, leq_out = self.best_split_col(col_index,
                                                              min_samples_leaf = min_samples_leaf)
            
            splits.append(split)
            errors.append(mse)
            gr_node_out.append(gr_out)
            leq_node_out.append(leq_out)
            
            if self.evolution:
                val_above, val_below = self.split_validation(col_index,split)
                
                val_sum_res_sq_above = sum(val_above.iloc[:,-1].apply(lambda x : (x - gr_out)**2))
                val_sum_res_sq_below = sum(val_below.iloc[:,-1].apply(lambda x : (x - leq_out)**2))
                
                validation_sum_residual_squares.append(val_sum_res_sq_above + val_sum_res_sq_below)
                
        if self.evolution:
            
            output = sorted(zip(splits,
                                errors,
                                gr_node_out,
                                leq_node_out,
                                validation_sum_residual_squares),
                            key = lambda x : x[-1],
                            reverse = True)
            
        else:
            
            output = min(zip(splits,
                             errors,
                             gr_node_out,
                             leq_node_out),
                         key = lambda x : x[1])
            
        return output
        

class Tree:
    def __init__(self,nodes = None,changed = None):
        self.nodes = nodes
        self.changed = changed
        
        
    def next_gen(self,min_samples_split = 2,min_samples_leaf = 1):
        
        nodes_to_split = [el for el in self.nodes 
                          if el.feature is None
                          and len(el.train_df_slice) > min_samples_leaf]
                
        for node in nodes_to_split:
            
            splits,gr_node_out,leq_node_out = node.all_features_best_splits(min_samples_leaf)
            
            for i in range(len(splits)):
                
                val_above,val_below = split_df(node.val_df_slice,i,splits[i])
    
    
    def grow_specific(self,train_df,validation_df,features,min_samples_split = 2,min_samples_leaf = 1):
        if type(features) is not list:
            features = [features]
        
        self.changed = False
        for feature in features:
            #print('Build node for ',feature)
            if self.nodes is None:
                
                split,mse = best_split_col(train_df,feature,True)
                train_above,train_below = split_df(train_df,feature,split)
                validation_above,validation_below = split_df(validation_df,feature,split)
                
                self.nodes = [Node(index = 0,
                                   feature = feature,
                                   threshold = split,
                                   n_samples = len(train_df),
                                   impurity = mse,
                                   output_value = mean(train_df.iloc[:,-1]),
                                   child_left = 1,
                                   child_right = 2),
                              Node(index = 1,
                                   train_df_slice = train_below,
                                   val_df_slice = validation_below,
                                   output_value = mean(train_below.iloc[:,-1])),
                              Node(index = 2,
                                   train_df_slice = train_above,
                                   val_df_slice = validation_above,
                                   output_value = mean(train_above.iloc[:,-1]))
                              ]
                
                self.changed = True
                
            else:
                nodes_to_split = [el for el in self.nodes 
                                  if el.feature is None 
                                  and el.train_df_slice.iloc[:,feature].nunique() >= min_samples_split]
                
                for node in nodes_to_split:
                        
                    self.changed = True
                    
                    split,mse = best_split_col(node.train_df_slice,feature,True,min_samples_leaf)
                    
                    node.feature = feature
                    node.threshold = split
                    node.impurity = mse
                    
                    train_above,train_below = split_df(node.train_df_slice,feature,split)
                    
                    validation_above,validation_below = split_df(node.val_df_slice,feature,split)
                    
                    node.child_left = len(self.nodes)
                    self.nodes.append(Node(index = len(self.nodes),
                                           train_df_slice = train_below,
                                           val_df_slice = validation_below,
                                           output_value = mean(train_below.iloc[:,-1])))
                    
                    
                    node.child_right = len(self.nodes)
                    self.nodes.append(Node(len(self.nodes),
                                           train_df_slice = train_above,
                                           val_df_slice = validation_above,
                                           output_value = mean(train_above.iloc[:,-1])))
                
                    node.train_df_slice = None
                    node.val_df_slice = None
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