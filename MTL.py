 # coding: utf-8


import numpy as np
import pandas as pd
import RF as rf
import math as Math



def normal_pro(mean, std, value):
    if std == 0 :
        if mean == value:
            pro =1
        else:
            pro=0.00000001
    else:
        pro = np.exp(-(value - mean) ** 2 /(2* std **2))/(Math.sqrt(2*Math.pi)* std)
    return pro



def covarite_loss(source_list, target, feature_list, value_list):
    source_num = len(source_list)
    split_dict = {}
    
    for feature, value in zip(feature_list, value_list):
     
        split_dict[feature]= 0
        i = 0
        
        while i < source_num:
            sub_source = source_list[i]
          
            target_feature = target.iloc[:, feature]
            source_feature = sub_source.iloc[:, feature]
            mean_t = np.mean(target_feature)
            std_t = np.std(target_feature)
            mean_s = np.mean(source_feature)
            std_s = np.std(source_feature)
            pro_t = normal_pro(mean_t,std_t,value)
            pro_s = normal_pro(mean_s,std_s,value)
            split_dict[feature] += (1 + np.exp(pro_s)) / ((1 + np.exp(pro_t)))
            i+=1
        split_dict[feature] = split_dict[feature] / source_num
    
    return split_dict
                



def mmd(source, target, feature):
    k = 0
    source_x = pd.DataFrame()
    for s in source:
        if k > 0:
            source_x.append(pd.DataFrame(s))
        else:
            source_x = pd.DataFrame(s)
        k += 1 
    
    source_x = source_x.iloc[:,feature].tolist()
    target_x = list(target.iloc[:,feature])
    
    
    def kernel(x, y):
        sigma = 0.5
        return np.exp(-(x ** 2 + y ** 2) / sigma)
    
    m = len(source_x)
    n = len(target_x)
    
    mmd_distance = 1 / (m * n)
    kernel_sum = 0
    for i in range(0,m-1):
        for j in range(0,n-1):
            kernel_sum += kernel(source_x[i], target_x[j])
            j += 1
        i += 1
    
    mmd_distance = kernel_sum / (m * n) 
    
    return mmd_distance



def multi_loss(source_list, target, feature_list, value_list):
    mtl_split_dict = covarite_loss(source_list, target, feature_list, value_list)
    alpha=1
    total_sample_num = 0
    total_label = list()
    source_feature = list()
    
    for s in source_list:
        total_sample_num += s.iloc[:,0].size
        total_label += list(float(row[-1]) for idx,row in s.iterrows())
        
        source_feature.append(s)
                            
    target = pd.DataFrame(target)
    total_sample_num += target.iloc[:,0].size
    target_num = target.iloc[:,0].size
    
    target_label = list(float(row[-1]) for idx,row in target.iterrows())
    total_label += list(float(row[-1]) for idx,row in target.iterrows())
    
    regr = np.sum(total_label)/ total_sample_num
    
    for key in mtl_split_dict:
        cov_loss = mtl_split_dict[key]
        mtl_split_dict[key] = (np.sum(target_label) + alpha * regr * cov_loss) / (target_num + alpha)
    
    feature_mtl = max(mtl_split_dict, key = mtl_split_dict.get)
    
    
    lamda = np.exp(-mmd(source_feature, target, feature_mtl))
    value_list = list(value_list)
    mtl_split = 0
    for key in mtl_split_dict:
        if float(key) == feature_mtl:
            mtl_split = mtl_split_dict[key]
    
    i = 0
    v_idex = 0
    
    if i <len(feature_list):
        if int(feature_list[i])==feature_mtl:
            v_idex=i
        i+=1
        
    value_mtl = (1-lamda) * value_list[int(v_idex)] + lamda * mtl_split
    
    return feature_mtl, value_mtl
