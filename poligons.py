# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 20:03:27 2019

@author: 106300
"""

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def polygon_under_graph(xlist, ylist):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make verts a list, verts[i] will be a list of (x,y) pairs defining polygon i
verts = []

# Set up the x sequence
xs = np.linspace(0., 10., 2)

# The ith polygon will appear on the plane y = zs[i]
zs = range(1)

for i in zs:
    ys = np.random.rand(len(xs))
    verts.append(polygon_under_graph(xs, ys))

verts2 = [[(0.0, 0.0),
           (0.0, 1),
           (10.0, 1),
           (10.0, 0.0)]]
poly = PolyCollection(verts2, facecolors=['r'], alpha=.6)
ax.add_collection3d(poly, zs=zs, zdir='y')
verts2 = [[(0.0, 0.0),
           (0.0, 1),
           (1.0, 1),
           (1.0, 0.0)]]


poly = PolyCollection(verts2, facecolors=['g'], alpha=.6)
ax.add_collection3d(poly, zs=zs, zdir='x')
verts2 = [[(0.0, 0.0),
           (0.0, 1),
           (10.0, 1),
           (10.0, 0.0)]]
poly = PolyCollection(verts2, facecolors=['b'], alpha=.6)
ax.add_collection3d(poly, zs=zs, zdir='z')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(0, 10)
ax.set_ylim(-1, 4)
ax.set_zlim(0, 1)

plt.show()