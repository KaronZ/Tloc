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




def js_gain(p,q):
    p = np.array(p)
    q = np.array(q)
    fit_length = 0
    if len(p) >= len(q):
        fit_length = len(q)
    else:
        fit_length = len(p)
    
    p_new = p[0:fit_length]
    q_new = q[0:fit_length]
    
   
    M = (p_new + q_new)/2
    
    js = 0.5 * scipy.stats.entropy(p_new, M) + 0.5 * scipy.stats.entropy(q_new, M)
    return js
 




def loss(left, right, t_l, t_r):
    loss_left = 0
    loss_right =0
    for t in left:
        loss_left += (t - np.mean(t_l)) ** 2
    for t in right:
        loss_right += (t-np.mean(t_r)) ** 2
        
    if len(left) == 0:
        loss_left = -loss_left
    else:
        loss_left = -loss_left / len(left)
        
    if len(right) == 0:
        loss_right = -loss_right
    else:
        loss_right = -loss_right / len(right)
    loss = loss_left + loss_right
    return loss
    



def optimize(all_data, target_data, feature_mtl, value_mtl):
    target = np.array(target_data)#np.ndarray()
    target = target.tolist()#list
    #target_data=list(target_data)
   
    left, right = rf.data_spilt(all_data, int(feature_mtl), value_mtl)
    left_t, right_t = rf.data_spilt(target, int(feature_mtl), value_mtl, True)
    all_label_left = list(float(row[-1]) for row in left)
    target_label_left = list(float(row[-1]) for row in left_t)
    all_label_right = list(float(row[-1]) for row in right)
    target_label_right = list(float(row[-1]) for row in right_t)
    
    theta=np.std(target_data.iloc[:,feature_mtl])
    
    divergence_gain = -999
    
    value_str = value_mtl
    i = value_mtl-theta
    
    while i <= (value_mtl + theta):
        left_n, right_n = rf.data_spilt(all_data, feature_mtl, i)
        left_n_t, right_n_t = rf.data_spilt(target, feature_mtl, i)
        
        all_n_left = list(float(row[-1]) for row in left_n)
        target_n_left = list(float(row[-1]) for row in left_n_t)
        all_n_right = list(float(row[-1]) for row in right_n)
        target_n_right = list(float(row[-1]) for row in right_n_t)
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
        
    






