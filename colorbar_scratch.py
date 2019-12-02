#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:15:43 2019

@author: instalador
"""


import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from collections import OrderedDict


def Reverse_ColorBar(Bar_input):
    Bar_output = np.zeros(Bar_input.shape)
    for k in range(Bar_input.shape[0]):
        Bar_output[k,:] = Bar_input[Bar_input.shape[0]-1-k,:] 
    return Bar_output


#------------------------------------------------------------------------------
def ColorMap():
    cmp_list    = [ 'viridis','Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                    'YlOrBr' , 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    c_map        = cm.get_cmap(cmp_list[0]+'_r', 256) # ''= normal, '_r' = reversed
    color_BAR   = c_map(np.linspace(0, 1, 64))
    
    matlab      = 'inicial'    #-----------el de Matlab---------------
    matlab      = 'new'
    if matlab == 'inicial': 
        color_BAR_r = Reverse_ColorBar(color_BAR)
        c_map = ListedColormap(color_BAR_r)  
    return c_map,color_BAR,
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def barra_RGB(color_BAR):
#    'rgb(0.9553  , 0.901065, 0.118128)'
    color_BAR_RGB = []
    col_scale = []
    for counter,color in enumerate(color_BAR):
        color_RGB = 'rgb('+str(color[0])+','+str(color[1])+','+str(color[2])+')'
        color_BAR_RGB.append(color_RGB)
        col_scale.append([counter,color_RGB])
    
    z = np.arange(64)
    
    return color_BAR_RGB,z,col_scale
    
c_mapa,color_barra = ColorMap()
RGB_bar,Z,popo  = barra_RGB(color_barra)



pepe = [
        # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
        [0, "rgb(0, 100, 0)"],
        [0.1, "rgb(0, 100, 0)"],

        # Let values between 10-20% of the min and max of z
        # have color rgb(20, 20, 20)
        [0.1, "rgb(20, 20, 20)"],
        [0.2, "rgb(20, 20, 20)"],

        # Values between 20-30% of the min and max of z
        # have color rgb(40, 40, 40)
        [0.2, "rgb(40, 40, 40)"],
        [0.3, "rgb(40, 40, 40)"],

        [0.3, "rgb(60, 60, 60)"],
        [0.4, "rgb(60, 60, 60)"],

        [0.4, "rgb(80, 80, 80)"],
        [0.5, "rgb(80, 80, 80)"],

        [0.5, "rgb(100, 100, 100)"],
        [0.6, "rgb(100, 100, 100)"],

        [0.6, "rgb(120, 120, 120)"],
        [0.7, "rgb(120, 120, 120)"],

        [0.7, "rgb(140, 140, 140)"],
        [0.8, "rgb(140, 140, 140)"],

        [0.8, "rgb(160, 160, 160)"],
        [0.9, "rgb(160, 160, 160)"],

        [0.9, "rgb(180, 180, 180)"],
        [1.0, "rgb(180, 180, 180)"]
    ]


fig = go.Figure()

fig.add_trace(go.Heatmap(
    z=[Z],
    colorscale=popo,
    colorbar=dict(
        tick0=0,
        dtick=1
    )
))

# Add shapes
p_x = 2
p_y = 0
fig.add_shape(go.layout.Shape(type="rect",x0=p_x-0.25,y0=p_y-0.25,x1=p_x+0.25,y1=p_y+0.25,
                              line=dict(color=RGB_bar[0]),
                              fillcolor=RGB_bar[0]),
                            )
    
    
fig.update_xaxes(range=[0, 15], showgrid=True)
fig.update_yaxes(range=[0, 1])
fig.show()