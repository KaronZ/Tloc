#-*- coding: utf-8 -*-
import csv
from random import seed
from random import randrange
from math import sqrt


def loadCSV(filename):
    dataSet = []
    with open(filename, 'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet


def column_to_float(dataSet):
    featLen = len(dataSet[0])-1
    for data in dataSet:
        for column in range(featLen):
            data[column] = float(data[column].strip())


def spiltDataSet(dataSet, n_folds):
    fold_size = int(len(dataSet) / n_folds)
    dataSet_copy = list(dataSet)
    dataSet_spilt=[]
    for i in range(n_folds):
        fold=[]
        while len(fold) < fold_size:  
            index = randrange(len(dataSet_copy))
            fold.append(dataSet_copy.pop(index))  
        dataSet_spilt.append(fold)
    return dataSet_spilt

def get_subsample(dataSet, ratio):
    subdataSet = []
    lenSubdata = round(len(dataSet) * ratio)
    while len(subdataSet) < lenSubdata:
        index = randrange(len(dataSet)-1)
        subdataSet.append(dataSet[index])
    #print len(subdataSet)
    return subdataSet

def data_spilt(dataSet, index, value):
    left = []
    right = []
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
	
def spilt_loss(left, right, class_values=None, is_classification=False):
    loss = 0.0
    if is_classification == True:
        for class_value in class_values:
            left_size = len(left)
            if left_size != 0:  
                prop = [row[-1] for row in left].count(class_value) / float(left_size)
                loss += (prop * (1.0 - prop))
            right_size=len(right)
            if right_size != 0:
                prop = [row[-1] for row in right].count(class_value) / float(right_size)
                loss += (prop*(1.0 - prop))
    else:
        sub_left = list(row[-1] for row in left)
        sub_right = list(row[-1] for row in right)
        node_sum = sum(sub_left)+sum(sub_right)
        averge = node_sum /(len(left)+len(right))
        node = list()
        node.append(left)
        node.append(right)
        
        for child in node:
            child_size = len(child)
            if size == 0:
                continue
            child_label = list(record[-1] for record in child)
            for label in group_label:
                reg += (label-average)^2
            reg = float(reg) / size
            reg = size * reg / number_node
            loss += reg
        
    return loss


def get_best_spilt(dataSet,n_features):
    features = []
    class_values = list(set(row[-1] for row in dataSet))
    b_index, b_value, b_loss, b_left, b_right = 999, 999, 999, None, None
    while len(features) < n_features:
        index = randrange(len(dataSet[0])-1)
        if index not in features:
            features.append(index)
    #print 'features:',features
    for index in features:
        for row in dataSet:
            left,right = data_spilt(dataSet,index,row[index])
            loss = spilt_loss(left,right,class_values)
            if loss < b_loss:
                b_index,b_value,b_loss,b_left,b_right=index,row[index],loss,left,right
    #print b_loss
    #print type(b_index)
    return {'index':b_index,'value':b_value,'left':b_left,'right':b_right}

def get_best_spilt_candidate(dataSet, n_features):
    features = []
    class_values = list(set(row[-1] for row in dataSet))
    b_loss_dict = {}
    
    while len(features) < n_features:
        index = randrange(len(dataSet[0])-1)
        if index not in features:
            features.append(index)
    #print 'features:',features
    for index in features:
        b_loss_dict[index] = 999
        for row in dataSet:
            left,right = data_spilt(dataSet, index, row[index])
            loss = spilt_loss(left, right, class_values)
            if loss < b_loss_dict[index]:
                b_loss_dict[index] = loss
   
    return b_loss_dict
    

def decide_label(data):
    output=[row[-1] for row in data]
    return max(set(output), key = output.count)

def sub_spilt(root,n_features,max_depth,min_size,depth):
    left=root['left']
    #print left
    right=root['right']
    del(root['left'])
    del(root['right'])
    #print depth
    if not left or not right:
        root['left'] = root['right'] = decide_label(left + right)
        #print 'testing'
        return
    if depth > max_depth:
        root['left'] = decide_label(left)
        root['right'] = decide_label(right)
        return
    if len(left) < min_size:
        root['left'] = decide_label(left)
    else:
        root['left'] = get_best_spilt(left, n_features)
        #print 'testing_left'
        sub_spilt(root['left'], n_features, max_depth, min_size, depth+1)
    if len(right) < min_size:
        root['right'] = decide_label(right)
    else:
        root['right'] = get_best_spilt(right, n_features)
        #print 'testing_right'
        sub_spilt(root['right'], n_features, max_depth, min_size, depth+1)    

def build_tree(dataSet, n_features, max_depth, min_size):
    root = get_best_spilt(dataSet, n_features)
    sub_spilt(root, n_features, max_depth, min_size, 1) 
    return root

def build_transfer_tree(dataSet, split_index, split_value, max_depth, min_size):
    left, right = data_spilt(dataSet, split_index, split_value)
    root = {'index':split_index,'value':split_value,'left':left,'right':right}
    sub_spilt(root,n_features, max_depth, min_size, 1) 
    return root

def predict(tree, row):
    predictions = []
    if row[tree['index']] < tree['value']:
        if isinstance(tree['left'],dict):
            return predict(tree['left'],row)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'],dict):
            return predict(tree['right'],row)
        else:
            return tree['right']

def bagging_predict(trees, row, isclassification=False):
    predictions = [predict(tree, row) for tree in trees]  
    if isclassification:
        return max(set(predictions), key=predictions.count)
    else:
        return sum(predictions)/len(predictions) 


def random_forest(train, test, ratio, n_features, max_depth, min_size, n_trees):
    trees = []
    for i in range(n_trees):
        train = get_subsample(train,ratio)
        tree = build_tree(train,n_features,max_depth,min_size)
        #print 'tree %d: '%i,tree
        trees.append(tree)
    #predict_values = [predict(trees,row) for row in test]
    predict_values = [bagging_predict(trees, row) for row in test]
    return predict_values


def accuracy(predict_values, actual, is_classification=False):
    correct = 0.0
    score = 0.0
    if is_classification==True:
        for i in range(len(actual)):
            if actual[i]==predict_values[i]:
                correct+=1
        score = correct/float(len(actual))
    else:
        for i in range(len(actual)):
            error = abs(actual[i] - predict_values[i])
            correct += error
        score = np.median(correct)
    return score
            
        

        
        
            
        

