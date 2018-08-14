
# coding: utf-8


import csv
from random import seed
from random import randrange
from math import sqrt
import RF as rf
import MTL as mtl
import numpy as np
import pandas as pd
import scipy.stats
import STRUT as strut


def weight_loss(w_m, w_r, target, feature_mtl, value_mtl, value_str):
    left_m, right_m = rf.data_spilt(target, feature_mtl, value_mtl)
    loss_m = rf.spilt_loss(left,right)
	
    left_r, right_r = rf.data_spilt(target, feature_mtl, value_str)
    loss_r = rf.spilt_loss(left,right)
	
    beta = 0.8
	
    w_m_t = w_m * (beta ** loss_m)
    w_r_t = w_m * (beta ** loss_r)
	
    weight_m = w_m_t / (w_m_t + w_r_t)
    weight_r = w_r_t / (w_m_t + w_r_t)
	
    return weight_m, weight_r
	

def transferForest(train, targetID, n_features, max_depth, min_size, n_trees):
    transfer_trees =[]
    
    train = pd.DataFrame(train.sort_values(by="BS_ID"))
    
    group_list = list(train.iloc[:,0]) # The default first column is the primary base station identifier
    group_list = sorted(group_list)
    source_list = []
    for group in group_list:
        if group != targetID:
            source_list.append(train[train['BS_ID'] == group])
    target = train[train['BS_ID']] = targetID
    
    feature_list = []
    value_list = []

	w_m = 0.5
	w_r = 0.5
    
    for i in range(n_trees):
        normal_tree = rf.get_best_spilt_candidate(dataSet,n_features)
        
        for key in normal_tree:
            feature_list.append(key)
            value_list.append(normal_tree[key])
            
        feature_mtl, value_mtl = mtl.multi_loss(source_list, target, feature_list, value_list)
        value_str = strut.optimize(train, target, feature_mtl, value_mtl)
        
        spilt_value = w_m * value_mtl + w_r * value_str
		w_m, w_r = weight_loss(w_m, w_r, target, feature_mtl, value_mtl, value_str)
		
        tree = rf.build_transfer_tree(train, feature_mtl, split_value, max_depth, min_size)
        
        transfer_trees.append(tree)
    
    return transfer_trees



def transfer_forest_predict(train_model, test):
    predict_values = [rf.bagging_predict(train_model, row) for row in test]
    return predict_values

