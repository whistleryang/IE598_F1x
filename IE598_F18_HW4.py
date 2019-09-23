#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:41:12 2019

@author: whistler
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/whistler/Desktop/MachineLearning/housing.csv', header=0)
df = df.dropna()

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
plt.figure()
sns.set(font_scale=1)
hm = sns.heatmap(corr_coef, 
                 cbar=True,
                 annot=True, 
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 7},
                 yticklabels=df.columns.values,
                 xticklabels=df.columns.values)
plt.rcParams['figure.dpi'] = 600
plt.show()

#spilt data
from sklearn.model_selection import train_test_split
X = df.drop('MEDV', axis = 1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

reg = LinearRegression()
reg.fit(X_train, y_train)

y_train_reg_pred = reg.predict(X_train)
y_test_reg_pred = reg.predict(X_test)

plt.figure()
plt.scatter(y_train_reg_pred,  y_train_reg_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_reg_pred,  y_test_reg_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.rcParams['figure.dpi'] = 100
plt.show()

print('linear regression coef: ', reg.coef_)
print('linear regression intercept: ', reg.intercept_)

print('linear regression MSE train: ', mean_squared_error(y_train, y_train_reg_pred),
      '   test: ', mean_squared_error(y_test, y_test_reg_pred))

print('linear regression R^2 train: ', r2_score(y_train, y_train_reg_pred),
      '   test: ', r2_score(y_test, y_test_reg_pred))

#Ridge regressio
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_train_ridge_pred = ridge.predict(X_train)
y_test_ridge_pred = ridge.predict(X_test)

plt.figure()
plt.rcParams['figure.dpi'] = 100
plt.scatter(y_train_ridge_pred,  y_train_ridge_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_ridge_pred,  y_test_ridge_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print('Ridge regressio coef: ', ridge.coef_)
print('Ridge regressio intercept: ', ridge.intercept_)
    
print('Ridge regressio MSE train: ', mean_squared_error(y_train, y_train_ridge_pred),
          '   test: ', mean_squared_error(y_test, y_test_ridge_pred))
    
print('Ridge regressio R^2 train: ', r2_score(y_train, y_train_ridge_pred),
          '   test: ', r2_score(y_test, y_test_ridge_pred))

#alpha with best performance
ridge_alpha = []
ridge_coef = []
ridge_intercept = []
ridge_mse = []
ridge_r2 = []
for alpha in np.arange(0.1, 2, 0.1):
    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    
    y_train_ridge_pred = ridge.predict(X_train)
    y_test_ridge_pred = ridge.predict(X_test)
    
    ridge_alpha.append(alpha)
    ridge_coef.append(ridge.coef_)
    ridge_intercept.append(ridge.intercept_)
    
    ridge_mse.append(np.array([mean_squared_error(y_train, y_train_ridge_pred),
                      mean_squared_error(y_test, y_test_ridge_pred)]))
    
    ridge_r2.append(np.array([r2_score(y_train, y_train_ridge_pred),
                     r2_score(y_test, y_test_ridge_pred)]))



#Lasso regressio
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

y_train_lasso_pred = lasso.predict(X_train)
y_test_lasso_pred = lasso.predict(X_test)

plt.figure()
plt.rcParams['figure.dpi'] = 100
plt.scatter(y_train_lasso_pred,  y_train_lasso_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_lasso_pred,  y_test_lasso_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()

print('Lasso regressio coef: ', lasso.coef_)
print('Lasso regressio intercept: ', lasso.intercept_)

print('Lasso regressio MSE train: ', mean_squared_error(y_train, y_train_lasso_pred),
      '   test: ', mean_squared_error(y_test, y_test_lasso_pred))

print('Lasso regressio R^2 train: ', r2_score(y_train, y_train_lasso_pred),
      '   test: ', r2_score(y_test, y_test_lasso_pred))

#alpha with best performance
lasso_alpha = []
lasso_coef = []
lasso_intercept = []
lasso_mse = []
lasso_r2 = []
for alpha in np.arange(0.1, 2, 0.1):
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    
    y_train_lasso_pred = lasso.predict(X_train)
    y_test_lasso_pred = lasso.predict(X_test)
    
    lasso_alpha.append(alpha)
    lasso_coef.append(lasso.coef_)
    lasso_intercept.append(lasso.intercept_)
    
    lasso_mse.append(np.array([mean_squared_error(y_train, y_train_lasso_pred),
                      mean_squared_error(y_test, y_test_lasso_pred)]))
    
    lasso_r2.append(np.array([r2_score(y_train, y_train_lasso_pred),
                     r2_score(y_test, y_test_lasso_pred)]))

    
print("My name is Taiyu Yang")
print("My NetID is: taiyuy2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")