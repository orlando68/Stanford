#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:29:09 2019

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
def barra_RGB(color_BAR):
#    'rgb(0.9553  , 0.901065, 0.118128)'
    color_BAR_RGB = []
    for color in color_BAR:
        color_RGB = 'rgb('+str(color[0])+','+str(color[1])+','+str(color[2])+')'
        color_BAR_RGB.append(color_RGB)
        
    return color_BAR_RGB
    
c_mapa,color_barra = ColorMap()
RGB_bar            = barra_RGB(color_barra)


#mia = pd.DataFrame(data = np.zeros((10,3)),columns = ["eje x","eje y",'Number of points per Pixel'] )
#mia = pd.DataFrame(data = np.random.randn(10,3),columns = ["eje x","eje y",'Number of points per Pixel'] )
#fig = px.scatter(mia, x="eje x", y="eje y", color="Number of points per Pixel", color_continuous_scale=px.colors.sequential.Viridis)

fig = go.Figure()



fig.add_trace(go.Scatter(x=[1.5],y=[0.75],text=["Unfilled Rectangle"], mode="text"))


# Add shapes
fig.update_layout(shapes=[go.layout.Shape(type="path",path=" M 4,4 L 1,8 L 3,9 L3,8 L4,6 L4,5 Z",
            fillcolor=RGB_bar[0],line_color=RGB_bar[0])])



# Set axes properties
fig.update_xaxes(range=[0, 10], showgrid=True)
fig.update_yaxes(range=[0, 10])

# Add shapes
p_x = 2
p_y = 2
fig.add_shape(go.layout.Shape(type="rect",x0=p_x-0.25,y0=p_y-0.25,x1=p_x+0.25,y1=p_y+0.25,
                              line=dict(color=RGB_bar[0]),
                              fillcolor=RGB_bar[0]),
                            )
#fig.update_shapes(dict(xref='x', yref='y'))




fig.show()


