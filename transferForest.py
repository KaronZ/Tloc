# coding: utf-8

# In[2]:

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


# In[3]:

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

    for i in range(n_trees):
        normal_tree = rf.get_best_spilt_candidate(dataSet,n_features)
        
        for key in normal_tree:
            feature_list.append(key)
            value_list.append(normal_tree[key])
            
        feature_mtl, value_mtl = mtl.multi_loss(source_list, target, feature_list, value_list)
        value_str = strut.optimize(train, target, feature_mtl, value_mtl)
        
        spilt_value = 0.5 * value_mtl + 0.5 * value_str
        tree = rf.build_transfer_tree(train, feature_mtl, split_value, max_depth, min_size)
        
        transfer_trees.append(tree)
    
    return transfer_trees


# In[4]:

def transfer_forest_predict(train_model, test):
    predict_values = [rf.bagging_predict(train_model, row) for row in test]
    return predict_values
