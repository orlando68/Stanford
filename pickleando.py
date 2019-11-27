# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:46:40 2019

@author: 106300
"""

import pickle
import numpy as np
import pandas as pd

a = np.arange(10)
b = np.random.randn(10)
c = pd.DataFrame() 

# open a file, where you ant to store the data


file = open('important', 'wb')

pickle.dump(a, file)
pickle.dump(b,file)
pickle.dump(c,file)
# close the file
file.close()

file = open('important', 'rb')

data1 = pickle.load(file)
data2 = pickle.load(file)
data3 = pickle.load(file)

#file = open('objs.pkl', 'wb')
#for element in [a, b, c]:
#    pickle.dump(element, file)
#file.close()




#with open('objsa.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([a, b, c], f)
#
#with open('objsa.pkl','rb') as f:  # Python 3: open(..., 'rb')
#    df_stanford, x_lo_bnd, x_up_bnd, y_lo_bnd, y_up_bnd, HAL, epochs, n_cont, n_avail, n_fail1, i_diag1, n_fail2, n_fail3, seconds = pickle.load(f)
#    
file =  open('objs.pkl', 'rb') 
datos = pickle.load(file)
df_stanford, x_lo_bnd, x_up_bnd, y_lo_bnd, y_up_bnd, HAL, epochs, n_cont, n_avail, n_fail1, i_diag1, n_fail2, n_fail3, seconds = datos






