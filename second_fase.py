# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:09:13 2019

@author: 106300
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
def list_files(directory, extension):
    out = ([])
    files = os.listdir(directory)
#    print(files)
    for name in files:
        [_,ext]=name.split('.')
#        print(name)
        if ext == extension:
            #print (name)
            out= np.append(out,name)
            #print(out)
    return out
#---------------


path = 'C://OPG106300//PERSONAL//JustAnIlusion//GOOD//InputDataSets//scenario2//RX DATA//'
files = list_files(path,'csv')
path = 'C://OPG106300//PERSONAL//JustAnIlusion//GOOD//InputDataSets//scenario2_reduced//'

"""
file_name = 'motion_V1_10Hz_reduced.csv'
data = pd.read_csv(path + file_name) 
plt.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

# Prepare arrays x, y, z
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

x = data['Pos_X'].values
y = data['Pos_Y'].values
z = data['Pos_Z'].values
ax.plot(x, y, z, label='parametric curve')
ax.legend()
plt.show()
"""
file_name = 'MEAS_test_10Hz_PC_20190312_002_reduced.csv'
print(file_name)
data = pd.read_csv(path + file_name)
