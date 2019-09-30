#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:20:58 2019

@author: whistler
"""



#import data
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/whistler/Desktop/MachineLearning/hw5_treasury yield curve data.csv', header=0)
df = df.dropna()
df = df.iloc[:, 1:32]

#EDA
import seaborn as sns
import matplotlib.pyplot as plt

#summary statistics
pd.set_option('display.max_columns', None)
df.describe()

#pair plot
plt.figure()
sns.pairplot(df, height=2.5)
plt.tight_layout()
plt.show()

#heatmap
corr_coef = np.corrcoef(df.values.T)
plt.rcParams['figure.dpi'] = 200
plt.figure()
sns.set(font_scale=0.7)
hm = sns.heatmap(corr_coef, 
                 annot=False, 
                 square=True,
                 yticklabels=df.columns.values,
                 xticklabels=df.columns.values)
plt.show()
plt.rcParams['figure.dpi'] = 100

#spilt data
from sklearn.model_selection import train_test_split

X = df.iloc[:, 0:30]
y = df.iloc[:, 30]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.15, random_state=42)

#standard scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

#PCA
from sklearn.decomposition import PCA

#explained variance ratio for all
pca = PCA(n_components=None)
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature for all')
plt.ylabel('variance ratio')
plt.xticks(features)
plt.show()

#explained variance ratio for 3
pca = PCA(n_components = 3)
pca.fit(X_train_std)
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature for all')
plt.ylabel('variance ratio')
plt.xticks(features)
plt.show()


#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#for baseline
reg_base = LinearRegression()
reg_base.fit(X_train_std, y_train)

y_train_reg_base_pred = reg_base.predict(X_train_std)
y_test_reg_base_pred = reg_base.predict(X_test_std)

print('linear regression MSE for baseline   train: ', mean_squared_error(y_train, y_train_reg_base_pred),
      '   test: ', mean_squared_error(y_test, y_test_reg_base_pred))

print('linear regression R^2 for baseline   train: ', r2_score(y_train, y_train_reg_base_pred),
      '   test: ', r2_score(y_test, y_test_reg_base_pred))

#for PCA
reg_pca = LinearRegression()
reg_pca.fit(X_train_pca, y_train)

y_train_reg_pca_pred = reg_pca.predict(X_train_pca)
y_test_reg_pca_pred = reg_pca.predict(X_test_pca)

print('linear regression MSE for PCA   train: ', mean_squared_error(y_train, y_train_reg_pca_pred),
      '   test: ', mean_squared_error(y_test, y_test_reg_pca_pred))

print('linear regression R^2 for PCA   train: ', r2_score(y_train, y_train_reg_pca_pred),
      '   test: ', r2_score(y_test, y_test_reg_pca_pred))

#SVM regression
from sklearn.svm import SVR

#for baseline
svr_base = SVR(kernel='rbf', gamma=0.2, C=1.0)
svr_base.fit(X_train_std, y_train)

y_train_svr_base_pred = svr_base.predict(X_train_std)
y_test_svr_base_pred = svr_base.predict(X_test_std)

print('SVM regression MSE for baseline   train: ', mean_squared_error(y_train, y_train_svr_base_pred),
      '   test: ', mean_squared_error(y_test, y_test_svr_base_pred))

print('SVM regression R^2 for baseline   train: ', r2_score(y_train, y_train_svr_base_pred),
      '   test: ', r2_score(y_test, y_test_svr_base_pred))

#for PCA
svr_pca = SVR(kernel='rbf', gamma=0.2, C=1.0)
svr_pca.fit(X_train_pca, y_train)

y_train_svr_pca_pred = svr_pca.predict(X_train_pca)
y_test_svr_pca_pred = svr_pca.predict(X_test_pca)

print('SVM regression MSE for PCA   train: ', mean_squared_error(y_train, y_train_svr_pca_pred),
      '   test: ', mean_squared_error(y_test, y_test_svr_pca_pred))

print('SVM regression R^2 for PCA   train: ', r2_score(y_train, y_train_svr_pca_pred),
      '   test: ', r2_score(y_test, y_test_svr_pca_pred))