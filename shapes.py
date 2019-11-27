# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:34:40 2019

@author: 106300
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


plt.rcdefaults()


def label(xy, text):
    y = xy[1] - 0.15  # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)


fig, ax = plt.subplots()
# create 3x3 grid to plot the artists
#grid = np.mgrid[0.2:0.8:3j, 0.2:0.8:3j].reshape(2, -1).T

patches = []

# add a circle
#circle = mpatches.Circle(grid[0], 0.1, ec="none")
#patches.append(circle)
#label(grid[0], "Circle")

# add a rectangle
rect = mpatches.Rectangle( [0.025, 0.05], 0.05, 0.1, color =[0 , 0 , 0,1])
patches.append(rect)
#label(grid[1], "Rectangle")



collection = PatchCollection(patches)#, cmap=plt.cm.hsv, alpha=0.3)
#collection.set_array(np.array(colors))
ax.add_collection(collection)
#ax.add_line(line)

plt.axis('equal')
plt.axis('off')
plt.tight_layout()

plt.show()