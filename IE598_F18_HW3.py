#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:13:33 2019

@author: whistler
"""

import pandas as pd
import numpy as np

#read data
df = pd.read_csv('/Users/whistler/Desktop/MachineLearning/HY_Universe_corporate bond.csv', header=0)

#print the size of new data
obs_num = len(df.index)
attr_num = len(df.columns)
print('The number of observations is: ', obs_num)
print('The number of attributes is: ', attr_num)

#print the nature of attributes
attr_nature = df.dtypes
print('The nature of each attibute:')
print(attr_nature)

#statistical summaries for numercial attributes
summary = df.describe()
print('The statistical summaries for numercial attributes:')
print(summary)

#statistical summaries for categorical attributes
Moodys = {}
for value in df['Moodys']:
	Moodys[value] = Moodys.get(value, 0) + 1
print('summaries for Moodys:')
print(np.vstack(([key for key in Moodys.keys()], [value for value in Moodys.values()])))

S_and_P = {}
for value in df['S_and_P']:
	S_and_P[value] = S_and_P.get(value, 0) + 1
print('summaries for S_and_P:')
print(np.vstack(([key for key in S_and_P.keys()], [value for value in S_and_P.values()])))

Fitch = {}
for value in df['Fitch']:
	Fitch[value] = Fitch.get(value, 0) + 1
print('summaries for Fitch:')
print(np.vstack(([key for key in Fitch.keys()], [value for value in Fitch.values()])))

Bloomberg = {}
for value in df['Bloomberg Composite Rating']:
	Bloomberg[value] = Bloomberg.get(value, 0) + 1
print('summaries for Bloomberg Composite Rating:')
print(np.vstack(([key for key in Bloomberg.keys()], [value for value in Bloomberg.values()])))

#ecdf and qq-plot
from scipy import stats
import matplotlib.pyplot as plt

#function for ecdf
def ecdf(data):
    #remove 999
    data = list(filter(lambda x:x!=999, data))
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y
print('ecdf and qq-plot for Coupon:')
x_Coupon, y_Coupon = ecdf(df['Coupon'])
_ = plt.plot(x_Coupon, y_Coupon, marker = '.', linestyle = 'none')
_ = plt.xlabel('Coupon')
_ = plt.ylabel('ECDF')
plt.show()
stats.probplot(x_Coupon, dist="norm", plot=plt)
plt.show()

print('ecdf and qq-plot for Maturity At Issue months:')
x_Maturity, y_Maturity = ecdf(df['Maturity At Issue months'])
_ = plt.plot(x_Maturity, y_Maturity, marker = '.', linestyle = 'none')
_ = plt.xlabel('Maturity At Issue months')
_ = plt.ylabel('ECDF')
plt.show()
stats.probplot(x_Maturity, dist="norm", plot=plt)
plt.show()

print('ecdf and qq-plot for Issued Amount:')
x_Amount, y_Amount = ecdf(df['Issued Amount'])
_ = plt.plot(x_Amount, y_Amount, marker = '.', linestyle = 'none')
_ = plt.xlabel('Issued Amount')
_ = plt.ylabel('ECDF')
plt.show()
stats.probplot(x_Amount, dist="norm", plot=plt)
plt.show()

print('ecdf and qq-plot for LiquidityScore:')
x_Liquidity, y_Liquidity = ecdf(df['LiquidityScore'])
_ = plt.plot(x_Liquidity, y_Liquidity, marker = '.', linestyle = 'none')
_ = plt.xlabel('LiquidityScore')
_ = plt.ylabel('ECDF')
plt.show()
stats.probplot(x_Liquidity, dist="norm", plot=plt)
plt.show()

#visualizing interrelationships between attributes
#scatterplot
print('scatter plot of Maturity At Issue months and LiquidityScore:')
plt.plot(df['Maturity At Issue months'], df['LiquidityScore'], marker = '.', linestyle = 'none')
_ = plt.xlabel('Maturity At Issue months')
_ = plt.ylabel('LiquidityScore')
plt.show()
#Pearson's correlation
print('correlation coefficient of Maturity At Issue months and LiquidityScore:')
corr_coef = np.corrcoef(df['LiquidityScore'], df['Maturity At Issue months'])
print(corr_coef)

print("My name is Taiyu Yang")
print("My NetID is: taiyuy2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")