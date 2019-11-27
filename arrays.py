# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:11:22 2019

@author: 106300
"""
import numpy as np

a = np.array([2,3,1,4,5,3,4,3])
b = np.array([4,3,5,3,5,3,7,3])


a1 = np.where(a == 3)[0]
b1 = np.where(b == 3)[0]
c1 = np.intersect1d(a1, b1)
print(c1)
print(a[c1],b[c1])



X = np.array([[18, 3, 1 ,11], [8, 10, 11, 3],[ 9, 14, 6, 1],[ 4, 3, 15, 21]])
print(X)
K = X.T
col,row  = np.where( (X.T>0) & (X.T<10)) 
print('row',row+1)
print('col',col+1)
print(X[row,col])