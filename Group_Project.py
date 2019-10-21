#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:33:28 2019

@author: whistler
"""

import pandas as pd
import numpy as np

#read data
df = pd.read_csv('/Users/whistler/Desktop/MachineLearning/MLF_GP2_EconCycle.csv', header=0)

#export data to csv
def DataToCsv(file, data, columns):
    data = list(data)
    columns = list(columns)
    file_data = pd.DataFrame(data, index=range(len(data)), columns=columns)
    file_data.to_csv(file)

#EDA
import seaborn as sns
import matplotlib.pyplot as plt

#print the size of new data
obs_num = len(df.index)
attr_num = len(df.columns)
print('The number of observations is: ', obs_num)
print('The number of attributes is: ', attr_num)

#print the nature of attributes
attr_nature = df.dtypes
print('The nature of each attibute:')
print(attr_nature)
attr_nature.to_csv('/Users/whistler/Desktop/MachineLearning/Group_Project/attr_nature.csv')

#statistical summaries for numercial attributes
pd.set_option('display.max_columns', None)
summary = df.describe()
print('The statistical summaries for numercial attributes:')
print(summary)
summary.to_csv('/Users/whistler/Desktop/MachineLearning/Group_Project/summary.csv')

#pair plot
plt.figure()
sns.pairplot(df.iloc[:,1:17], height=2.5)
plt.tight_layout()
plt.show()

#heatmap
corr_coef = np.corrcoef(df.iloc[:,1:17].values.T)
plt.rcParams['figure.dpi'] = 200
plt.figure()
sns.set(font_scale=0.7)
hm = sns.heatmap(corr_coef, 
                 annot=False, 
                 square=True,
                 yticklabels=df.columns[1:17].values,
                 xticklabels=df.columns[1:17].values)
plt.show()
plt.rcParams['figure.dpi'] = 100

#spilt data
from sklearn.model_selection import train_test_split

X = df.iloc[:, 1:13]
y = df.iloc[:, 14:17]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.10, random_state=42)

y_train_3MO = y_train.iloc[:, 0]
y_train_6MO = y_train.iloc[:, 1]
y_train_9MO = y_train.iloc[:, 2]

y_test_3MO = y_train.iloc[:, 0]
y_test_6MO = y_train.iloc[:, 1]
y_test_9MO = y_train.iloc[:, 2]

#standard scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#PCA and explained variance ratio for all
from sklearn.decomposition import PCA

pca = PCA(n_components=None)
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
features = range(pca.n_components_)
plt.rcParams['figure.dpi'] = 200
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature for all')
plt.ylabel('variance ratio')
plt.xticks(features)
plt.show()
plt.rcParams['figure.dpi'] = 100

#keep 4 components for PCA
pca = PCA(n_components=4)
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#machine learning models: linear regression, SVM regressor, decision tree regressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

lr = LinearRegression()
svr = SVR()
dtr = DecisionTreeRegressor()

#hyperparameter tuning for basic models
from sklearn.model_selection import GridSearchCV

estimators = {'LinearRegression': lr, 'SupportVectorRegressor': svr, 'DecisionTreeRegressor': dtr}

params_grid_lr = {}
params_grid_svr = {'kernel': ['linear', 'rbf'], 'gamma':[0.01, 0.05, 0.1, 1, 5, 10], 'C': [0.01, 0.05, 0.1, 1, 5, 10]}
params_grid_dtr = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
params_grid = {'LinearRegression': params_grid_lr, 
               'SupportVectorRegressor': params_grid_svr, 
               'DecisionTreeRegressor': params_grid_dtr}

for key in estimators:
    grid = GridSearchCV(estimator = estimators[key], 
                        param_grid = params_grid[key], 
                        scoring = 'neg_mean_squared_error', 
                        n_jobs = -1, 
                        cv = 10)
    for i in range(3):
        grid.fit(X_train_pca, y_train.iloc[:, i])
        y_train_pred = grid.predict(X_train_pca)
        y_test_pred = grid.predict(X_test_pca)
        
        #best params, CV score, in sample score, out of sample score
        best_params = grid.best_params_
        CV_score = -grid.best_score_
        in_sample_score = -grid.score(X_train_pca, y_train.iloc[:, i])
        out_of_sample_score = -grid.score(X_test_pca, y_test.iloc[:, i])
        
        print("Result of fitting %d-month forward PCT on %s model:" %(3*i+3, key))
        print("      best params: ", best_params)
        print("      CV MSE: ", CV_score)
        print("      in sample MSE: ", in_sample_score)
        print("      out of sample MSE: ", out_of_sample_score)

#ensembling: random forest
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

#hyperparameter tuning for random forest
params_grid_rfr = {'n_estimators': [100, 200, 300, 400], 'max_depth': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
grid = GridSearchCV(estimator = rfr, 
                    param_grid = params_grid_rfr, 
                    scoring = 'neg_mean_squared_error', 
                    n_jobs = -1, 
                    cv = 10)

for i in range(3):
    grid.fit(X_train_pca, y_train.iloc[:, i])
    y_train_pred = grid.predict(X_train_pca)
    y_test_pred = grid.predict(X_test_pca)
    
    #best params, CV score, in sample score, out of sample score
    best_params = grid.best_params_
    CV_score = -grid.best_score_
    in_sample_score = -grid.score(X_train_pca, y_train.iloc[:, i])
    out_of_sample_score = -grid.score(X_test_pca, y_test.iloc[:, i])
    
    print("Result of fitting %d-month forward PCT on RandomForestRegressor model:" %(3*i+3))
    print("      best params: ", best_params)
    print("      CV MSE: ", CV_score)
    print("      in sample MSE: ", in_sample_score)
    print("      out of sample MSE: ", out_of_sample_score)
    
    #feature importance
    rfr.set_params(**best_params)
    rfr.fit(X_train_pca, y_train.iloc[:, i])
    importances = rfr.feature_importances_
    
    print('      feature importances: ', importances)