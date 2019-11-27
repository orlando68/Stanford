# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 00:12:46 2019

@author: 106300
"""


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

#im = np.array(Image.open('stinkbug.png'), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.plot(np.arange(10),np.arange(10))

# Create a Rectangle patch
rect = patches.Rectangle((2,2),1,1,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()