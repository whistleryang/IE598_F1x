#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:25:56 2019

@author: whistler
"""

#import data and split to variables and target
import pandas as pd
df = pd.read_csv('/Users/whistler/Desktop/MachineLearning/ccdefault.csv', header=0)
X = df.iloc[:, 1:24]
y = df.iloc[:, 24]

#train test splits
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,
                                                    random_state = 42, stratify = y)

#random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion = 'gini',max_depth = 3, random_state = 1)

#CV: individual fold scores
from sklearn.model_selection import cross_val_score

for i in [100, 200, 300, 400]:
    rf.set_params(n_estimators = i)
    print('n_estimators = ', i)
    scores_cv = cross_val_score(rf, X_train, y_train, cv = 10)
    print("CV individual fold scores: ")
    print(scores_cv)
    print("mean: ", scores_cv.mean())
    print("std: ", scores_cv.std())
    #CV: out of sample score
    rf.fit(X_train, y_train)
    scores_cv_out = rf.score(X_test, y_test)
    print("CV out of sample score: ", scores_cv_out)

#feature importance
rf.set_params(n_estimators = 300)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
print('n_estimators = 300')
print('feature importances: ')
print(importances)