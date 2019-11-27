# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:01:10 2019

@author: 106300
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap




newcolors =np.array([
                    [0.6980 , 0.6980 , 0.6980 , 1],
                    [0.6905 , 0.6905 , 0.6905 , 1],
                    [0.6830 , 0.6830 , 0.6830 , 1],
                    [0.6754 , 0.6754 , 0.6754 , 1],
                    [0.6679 , 0.6679 , 0.6679 , 1],
                    [0.6604 , 0.6604 , 0.6604 , 1],
                    [0.6528 , 0.6528 , 0.6528 , 1],
                    [0.6453 , 0.6453 , 0.6453 , 1],
                    [0.6378 , 0.6378 , 0.6378 , 1],
                    [0.6302 , 0.6302 , 0.6302 , 1],
                    [0.6227 , 0.6227 , 0.6227 , 1],
                    [0.6152 , 0.6152 , 0.6152 , 1],
                    [0.6076 , 0.6076 , 0.6076 , 1],
                    [0.6001 , 0.6001 , 0.6001 , 1],
                    [0.5926 , 0.5926 , 0.5926 , 1],
                    [0.5850 , 0.5850 , 0.5850 , 1],
                    [0.5775 , 0.5775 , 0.5775 , 1],
                    [0.5700 , 0.5700 , 0.5700 , 1],
                    [0.5624 , 0.5624 , 0.5624 , 1],
                    [0.5549 , 0.5549 , 0.5549 , 1],
                    [0.5474 , 0.5474 , 0.5474 , 1],
                    [0.5398 , 0.5398 , 0.5398 , 1],
                    [0.5323 , 0.5323 , 0.5323 , 1],
                    [0.5247 , 0.5247 , 0.5247 , 1],
                    [0.5172 , 0.5172 , 0.5172 , 1],
                    [0.5097 , 0.5097 , 0.5097 , 1],
                    [0.5021 , 0.5021 , 0.5021 , 1],
                    [0.4946 , 0.4946 , 0.4946 , 1],
                    [0.4871 , 0.4871 , 0.4871 , 1],
                    [0.4795 , 0.4795 , 0.4795 , 1],
                    [0.4720 , 0.4720 , 0.4720 , 1],
                    [0.4645 , 0.4645 , 0.4645 , 1],
                    [0.4569 , 0.4569 , 0.4569 , 1],
                    [0.4494 , 0.4494 , 0.4494 , 1],
                    [0.4419 , 0.4419 , 0.4419 , 1],
                    [0.4343 , 0.4343 , 0.4343 , 1],
                    [0.4268 , 0.4268 , 0.4268 , 1],
                    [0.4193 , 0.4193 , 0.4193 , 1],
                    [0.4117 , 0.4117 , 0.4117 , 1],
                    [0.4042 , 0.4042 , 0.4042 , 1],
                    [0.3967 , 0.3967 , 0.3967 , 1],
                    [0.3891 , 0.3891 , 0.3891 , 1],
                    [0.3816 , 0.3816 , 0.3816 , 1],
                    [0.3741 , 0.3741 , 0.3741 , 1],
                    [0.3665 , 0.3665 , 0.3665 , 1],
                    [0.3590 , 0.3590 , 0.3590 , 1],
                    [0.3515 , 0.3515 , 0.3515 , 1],
                    [0.3439 , 0.3439 , 0.3439 , 1],
                    [0.3364 , 0.3364 , 0.3364 , 1],
                    [0.3289 , 0.3289 , 0.3289 , 1],
                    [0.3213 , 0.3213 , 0.3213 , 1],
                    [0.2966 , 0.2966 , 0.2966 , 1],
                    [0.2719 , 0.2719 , 0.2719 , 1],
                    [0.2472 , 0.2472 , 0.2472 , 1],
                    [0.2225 , 0.2225 , 0.2225 , 1],
                    [0.1977 , 0.1977 , 0.1977 , 1],
                    [0.1730 , 0.1730 , 0.1730 , 1],
                    [0.1483 , 0.1483 , 0.1483 , 1],
                    [0.1236 , 0.1236 , 0.1236 , 1],
                    [0.0989 , 0.0989 , 0.0989 , 1],
                    [0.0742 , 0.0742 , 0.0742 , 1],
                    [0.0494 , 0.0494 , 0.0494 , 1],
                    [0.0247 , 0.0247 , 0.0247 , 1], 
                    [0      , 0      , 0      , 1]
                                                ])
newcmp = ListedColormap(newcolors)
norm = cm.colors.Normalize(vmin=5, vmax=10)

x      = [5,6,3,4,5,6,7,8,9,3,4,7]
y      = [2,4,3,6,8,1,3,5,6,7,5,5]
s      = [123,12,423,234,234,24,123,124,123,234,54,23]
colors = [30,2,4,5,6,7,8,3,4,6,7,63]

w = 15
h = 10
d = 70
plt.figure(figsize=(w, h), dpi=d)

coor = np.array([[0,35,0],[0,35,35]])
p = plt.Polygon(coor.T,facecolor = 'W')
plt.gca().add_patch(p)
coor = np.array([[0,35,35],[0,35,0]])
p = plt.Polygon(coor.T,facecolor = [0.0247 , 0.5 , 0.0247 , 1])
plt.gca().add_patch(p)
coor = np.array([[35,50,50,35],[0,0,35,35]])
p = plt.Polygon(coor.T,facecolor = 'r')
plt.gca().add_patch(p)
coor = np.array([[35,50,50],[35,35,50]])
p = plt.Polygon(coor.T,facecolor = 'orange')
plt.gca().add_patch(p)
coor = np.array([[0,35,50,0],[35,35,50,50]])
p = plt.Polygon(coor.T,facecolor = 'y')
plt.gca().add_patch(p)

#plt.scatter(x,y,s = 100,c=colors,cmap = newcmp,edgecolor = 'black',marker ='s',linewidth = 1, alpha = 0.75)
plt.scatter(x,y,s = 1,c=colors,cmap = newcmp,marker ='s')
#plt.scatter(x,y,s = 1000,c=colors,cmap = newcmp,  marker =((-20,-20),(20,-20),(20,20),(-20,20))   )
#plt.plot(x,y,c=colors,cmap = newcmp,marker ='s', alpha = 1)
for counter,k in enumerate(colors):
#    plt.plot(x[counter],y[counter],c=k)
    p = plt.Rectangle((x[counter],y[counter]),1,1,facecolor =newcolors[k,:])
    plt.gca().add_patch(p)



#p = plt.Rectangle((1,1),2,1,facecolor =newcolors[10,:])
#plt.gca().add_patch(p)


cbar = plt.colorbar()
cbar.set_label('amplitude')

#cb1 = mpl.colorbar.ColorbarBase( plt,cmap=newcmp,
#                                norm=norm,
#                                orientation='vertical')

#cbar.set_clim(0, 2.0)
plt.grid(True)
plt.axis('scaled') #da un aspecto cudrado
plt.show()

