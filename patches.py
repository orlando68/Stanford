# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:39:42 2019

@author: 106300
"""
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

def patch(vtx,color):
#    fig = plt.figure()
#    ax1 = fig.add_subplot(211, projection='3d')
    x3 = vtx[0,0]
    y3 = vtx[0,1]
    z3 = vtx[0,2]
    dx = vtx[1,0] - vtx[0,0]
    dy = vtx[3,1] - vtx[0,1]
    dz = vtx[4,2] - vtx[0,2]
#    ax.bar3d(x3, y3, z3, dx, dy, dz,color = ['w'])
    ax.bar3d(x3, y3, z3, dx, dy, dz,color = color)
   
#    plt.show()    
        

uno=np.array([[0   , 7.5 , 0],
              [0.5 , 7.5 , 0],
              [0.5 , 8.0 , 0],
              [0   , 8.0 , 0],
              [0   , 7.5 , 1.3617],
              [0.5 , 7.5 , 1.3617],
              [0.5 , 8.0 , 1.3617],
              [ 0  , 8.0 , 1.3617]])
        
dos = np.array([[         0  ,  8.0000   ,      0],
    [0.5000   , 8.0000   ,      0],
    [0.5000  ,  8.5000   ,      0],
    [     0  ,  8.5000   ,      0],
     [    0   , 8.0000   , 1.3802],
    [0.5000  ,  8.0000 ,   1.3802],
    [0.5000  ,  8.5000 ,   1.3802],
    [     0  ,  8.5000 ,   1.3802        ]])
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', facecolor='w')
patch(uno,(0.6980  ,  0.6980  ,  0.6980))
patch(dos,(1 ,0.1 ,0.1))
ax.view_init(90, -90)

#x3 = [1,2]
#y3 = [5,8]
#z3 = np.zeros(2)
#
#dx = np.ones(2)
#dy = np.ones(2)
#dz = [1,2]
#
#ax1.bar3d(x3, y3, z3, dx, dy, dz,color = ['w','r'])
#
#
#ax1.set_xlabel('x axis')
#ax1.set_ylabel('y axis')
#ax1.set_zlabel('z axis')

#ax.grid(False)
#ax.xaxis.pane.set_edgecolor('black')
#ax.yaxis.pane.set_edgecolor('black')
#ax.xaxis.pane.fill = False
#ax.yaxis.pane.fill = False
#ax.zaxis.pane.fill = False

# Get rid of the panes
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Get rid of the spines
#ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
#ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
#ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
#
# Get rid of the ticks
#ax.set_xticks([]) 
#ax.set_yticks([]) 
#ax.set_zticks([])

plt.show()