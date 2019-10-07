#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:28:07 2019

@author: whistler
"""

#import data and split to variables and target
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/whistler/Desktop/MachineLearning/ccdefault.csv', header=0)
X = df.iloc[:, 1:24]
y = df.iloc[:, 24]

#classifiers
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='gini',max_depth=3, random_state=1)

#Random test train splits: in and out of sample scores
from sklearn.model_selection import train_test_split

scores_RandomSplit_in = []
scores_RandomSplit_out = []
for i in range(1, 11):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i, stratify = y)
    tree.fit(X_train, y_train)
    scores_RandomSplit_in.append(tree.score(X_train, y_train))
    scores_RandomSplit_out.append(tree.score(X_test, y_test))

scores_RandomSplit_in = np.array(scores_RandomSplit_in)
scores_RandomSplit_out = np.array(scores_RandomSplit_out)

print("RandomSplit in sample scores: ")
print( scores_RandomSplit_in)
print("mean: ", scores_RandomSplit_in.mean())
print("std: ", scores_RandomSplit_in.std())

print("RandomSplit out of sample scores: ")
print(scores_RandomSplit_out)
print("mean: ", scores_RandomSplit_out.mean())
print("std: ", scores_RandomSplit_out.std())

#CV: individual fold scores
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify = y)
scores_cv = cross_val_score(tree, X_train, y_train, cv = 10)

#CV: out of sample score
tree.fit(X_train, y_train)
scores_cv_out = tree.score(X_test, y_test)

print("CV individual fold scores: ")
print(scores_cv)
print("mean: ", scores_cv.mean())
print("std: ", scores_cv.std())

print("CV out of sample score: ", scores_cv_out)