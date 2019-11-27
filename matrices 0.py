# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:13:52 2019

@author: 106300
"""
import numpy as np


diagonal  = np.array([[1, 2, 3], [3, 4, 5], [1, 2, 3]])
B         = np.array([[1, 2], [3, 4], [8,9]])


C1 = np.matmul(B.T,diagonal)

print(C1)
C  = np.matmul(C1,B)
print('---------------')
print(C)
#print(np.matmul(B,A.T))
