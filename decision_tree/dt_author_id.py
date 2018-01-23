#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
def classify(features_train, labels_train,split_number):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=split_number)
    clf = clf.fit(features_train, labels_train)
    
    
    return clf
    
def accuracy(features_train, labels_train, features_test, labels_test, split_number):
    clf=classify(features_train, labels_train,split_number)
    pred=clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc 

split_number=40
acc_min_samples_split=accuracy(features_train, labels_train, features_test, labels_test, split_number)
print acc_min_samples_split
total_features=len(features_train)+len(features_test)
print {"Total features": round(total_features)}
#########################################################


