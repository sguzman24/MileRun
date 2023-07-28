#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:16:36 2023

@author: seth guzman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from os import chdir
chdir('/Users/vanessaguzman/Desktop/CIS111')

dataRUN = pd.read_csv('mileRecords.csv')

x = np.array(dataRUN["year"])
x = x-x[0]
y = np.array(dataRUN["time"])



plt.plot(y,x, 'xk')
plt.xlabel('year')
plt.ylabel('time in minutes')
plt.title('mile record time')

def fmin(param):
    k = param[0]
    r = param[1]
    e = y-r*y*(1-y/k)*x
    return(np.dot(e, e.T))
def f(y,K,r,x):
    return(r*y*(1-y/K)*x)

x0 = (3.1, .25)
opt = minimize(fmin, x0)
print(opt)
print(opt.x[0])

xaxis = np.linspace(0,90,num=32)

plt.plot(y,x, 'xk')
plt.xlabel('year')
plt.ylabel('time in minutes')
plt.title('mile record time')
plt.grid()
K = 3.6
r = -.27
timepredict= f(xaxis, K, r, y)

plt.plot(xaxis, timepredict, color = "red")








