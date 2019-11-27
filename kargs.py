# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:12:41 2019

@author: 106300
"""
import numpy as np
def myFun(arg1, *argv): 
    print ("First argument :", arg1)
    print(len(argv))
    print(argv[0])
    print(argv[1])
    for arg in argv: 
        print()
        print("Next argument through *argv :") 
        print(arg)
  
myFun(np.arange(1,10), np.random.randn(5,5),np.arange(1,10)) 