# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:28:09 2019

@author: 106300
"""

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Circle
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(v1,v2):
    """
    Calculates the rotation matrix that changes v1 into v2.
    """
    v1/=np.linalg.norm(v1)
    v2/=np.linalg.norm(v2)

    cos_angle=np.dot(v1,v2)
    d=np.cross(v1,v2)
    sin_angle=np.linalg.norm(d)

    if sin_angle == 0:
        M = np.identity(3) if cos_angle>0. else -np.identity(3)
    else:
        d/=sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array([[    0,  d[2],  -d[1]],
                      [-d[2],     0,  d[0]],
                      [d[1], -d[0],    0]], dtype=np.float64)

        M = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1,0,0), index)

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    M = rotation_matrix(normal,(0, 0, 1)) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta
    
fig = plt.figure()    
ax =fig.add_subplot(111, projection='3d', facecolor='w')

p = Circle((0,0), .2) #Add a circle in the yz plane
ax.add_patch(p)
pathpatch_2d_to_3d(p, z = 0.5, normal = 'x')
pathpatch_translate(p, (0, 0.5, 0))

p = Circle((0,0), .2, facecolor = 'r') #Add a circle in the xz plane
ax.add_patch(p)
pathpatch_2d_to_3d(p, z = 0.5, normal = 'y')
pathpatch_translate(p, (0.5, 1, 0))

p = Circle((0,0), .2, facecolor = 'g') #Add a circle in the xy plane
ax.add_patch(p)
pathpatch_2d_to_3d(p, z = 0, normal = 'z')
pathpatch_translate(p, (0.5, 0.5, 0))

for normal in product((-1, 1), repeat = 3):
    p = Circle((0,0), .2, facecolor = 'y', alpha = .2)
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z = 0, normal = normal)
    pathpatch_translate(p, 0.5)