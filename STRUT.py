# coding: utf-8

# In[3]:

import csv
from random import seed
from random import randrange
from math import sqrt
import RF as rf
import MTL as mtl
import numpy as np
import pandas as pd
import scipy.stats


# In[9]:

def js_gain(p,q):
    p = np.array(p)
    q = np.array(q)
    fit_length = 0
    if p >= q:
        fit_length = p
    else:
        fit_length = q
    
    p = p[0:fit_length]
    q = q[0:fit_length]
    
    M = (p + q)/2
    js = 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
    return js
 


# In[10]:

def loss(left, right, t_l, t_r):
    loss_left = 0
    loss_right =0
    for t in left:
        loss_left += (t - np.mean(t_l)) ** 2
    for t in right:
        loss_right += (t-np.mean(t_r)) ** 2
    loss = -loss_left/len(left) + -loss_right/len(right)
    return loss
    


# In[15]:

def optimize(all_data, target_data, feature_mtl, value_mtl):
    
    left, right = rf.data_spilt(all_data, feature_mtl, value_mtl)
    left_t, right_t = rf.data_spilt(target_data, feature_mtl, value_mtl)
    all_label_left = list(row[-1] for row in left)
    target_label_left = list(row[-1] for row in left_t)
    all_label_right = list(row[-1] for row in right)
    target_label_right = list(row[-1] for row in right_t)
    
    theta=np.std(target_data[feature_mtl])
    
    divergence_gain = -999
    
    value_str = value_mtl
    
    while i in range(value_mtl - theta, value_mtl + theta):
        left_n, right_n = rf.data_spilt(all_data, feature_mtl, i)
        left_n_t, right_n_t = rf.data_spilt(target_data, feature_mtl, i)
        
        all_n_left = list(row[-1] for row in left_n)
        target_n_left = list(row[-1] for row in left_n_t)
        all_n_right = list(row[-1] for row in right_n)
        target_n_right = list(row[-1] for row in right_n_t)
        loss_new = loss(all_label_left, all_label_right, target_label_left, target_label_right)
        loss_old = loss(all_n_left, all_n_right, target_n_left, target_n_right)
        
        if loss_new >= loss_old:
            weight_left = len(target_label_left) / (len(target_label_left) + len(target_label_right))
            weight_right = len(target_label_right) /(len(target_label_left) + len(target_label_right))
            divergence_tmp = 1 - weight_left * js_gain(all_label_left, all_n_left) - weight_right * js_gain(all_label_right, all_n_right)
            
            if divergence_tmp >= divergence_gain:
                divergence_gain = divergence_tmp
                value_str = i
        
        i += 0.05*theta
        
    return value_str
        
    
