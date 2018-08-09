# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import RF as rf
import math as Math


# In[2]:

def normal_pro(mean, std, value):
    pro = np.exp(-(value - mean) ** 2 /(2* std **2))/(Math.sqrt(2*Math.pi)* std)
    return pro


# In[3]:

def covarite_loss(source_list, target, feature_list, value_list):
    source_num = len(source_list)
    split_dict={}
    
    for feature,value in zip(feature_list, value_list):
        split_dict[feature]= 0
        i=0
        while i < source_num:
            sub_source = source_list[i]
            target_feature = target[feature]
            source_feature = sub_source[feature]
            mean_t = np.mean(target_feature)
            std_t = np.std(target_feature)
            mean_s = np.mean(source_feature)
            std_s = np.std(source_feature)
            pro_t = normal_pro(mean_t,std_t,value)
            pro_s = normal_pro(mean_s,std_s,value)
            split_dict[feature] += (1 + np.exp(pro_s)) / ((1 + np.exp(pro_t)))
            i+=1
        split_dict[feature] = split_dict[feature] / source_num
    
    return splict_dict
                


# In[6]:

def mmd(source, target, feature):
    source_x = source[feature]
    target_x = target[feature]
    sigma = 0.5
    
    def kernel(x,y, sigma):
        return np.exp(-(x ** 2 + y ** 2) / sigma)
    
    m = len(source_x)
    n = len(target_x)
    
    mmd_distance = 1 / (m * n)
    kernel_sum = 0
    while i in range(0,m-1):
        while j in range(0,n-1):
            kernel_sum += kernel(source_x[i], target[j])
            j += 1
        i += 1
    
    mmd_distance = kernel_sum / (m * n) 
    
    return mmd_distance


# In[7]:

def multi_loss(source_list, target, mtl_split_dict, value_list):
    alpha=1
    total_sample_num = 0
    total_label=list()
    source_feature = pd.DataFrame()
    
    for s in source_list:
        s = pd.DataFrame(s)
        total_sample_num += s.iloc[:,0].size
        total_label += list(row[-1] for row in s)
        source_feature.append(s)
                            
    target = pd.DataFrame(target)
    total_sample_num += target.iloc[:,0].size
    target_num = target.iloc[:,0].size
    
    target_label = list(row[-1] for row in target)
    total_label += list(row[-1] for row in target)
    
    regr = np.sum(total_label)/ total_sample_num
    
    for key in mtl_split_dict:
        cov_loss = mtl_split_dict[key]
        mtl_split_dict[key] = (np.sum(target_label) + alpha * regr * cov_loss) / (target_num + alpha)
    
    feature_mtl = max(mtl_split_dict, key = mtl_split_dict.get)
    lamda = np.exp(-mmd(source_feature, target, feature_mtl))
    value_mtl = (1-lamda) * value_list[feature_mtl] + lamda * mtl_split_dict[feature_mtl]
    
    return feature_mtl, value_mtl
    


# In[ ]:



