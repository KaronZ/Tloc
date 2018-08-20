# coding: utf-8


import csv
from math import sqrt
import RF as rf
import numpy as np
import pandas as pd
import transferForest as trans





if __name__=='__main__':
    
    dataSet, feature = rf.loadCSV('../../xxx.csv') #your file path, the file consists of source domain data and target domian data
    rf.column_to_float(dataSet)

    targetID = 'xxx' #specifiy the target domain  ID
    
    # parametres in random forests
    n_folds = 10
    max_depth = 15
    min_size = 1
    ratio = 1.0
   
    n_features = 35
    n_trees = 100
    
    #cross validation
    folds = rf.spiltDataSet(dataSet, n_folds)
    for fold in folds:
        train_set=folds[:]  
        train_set=sum(train_set,[]) 
        test_set=[]
        for row in fold:
            row_copy=list(row)
            row_copy[-1]=None
            test_set.append(row_copy)
        
        actual=[row[-1] for row in fold]
        
        
#====================train and test a transfer forest regression model=========================================      
        # train a transfer forest
        transfer_forest = trans.transferForest(train_set, targetID, n_features, max_depth, min_size, n_trees, feature)
        
        # make predictions test data
        predict_values_transfer = trans.transfer_forest_predict(transfer_forest, test_set)
        
        # compute error
        error = rf.accuracy(predict_values_transfer,actual)
        print (error)
        
#====================train and test a normal random froest regression model======================================
        # train a model and make predictions on test data
        predict_values = rf.random_forest(train_set, test_set, ratio, n_features, max_depth, min_size, n_trees)

        # compute error
        error = rf.accuracy(predict_values,actual)
        print (error)
