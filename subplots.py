#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:03:23 2019

@author: instalador
"""
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

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
    c_map        = cm.get_cmap(cmp_list[0]+'', 256) # ''= normal, '_r' = reversed
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
    
#------------------------------------------------------------------------------
def plot_ColorBar(dibujo,RGB_bar,Vref):
    D    = Vref/64
    d    = D/2
    
    for k in range(64):
        p_x = 1
        p_y = k*D+d
        fig.add_shape(go.layout.Shape(type="rect",x0=p_x-d,y0=p_y-d,x1=p_x+d,y1=p_y+d,
                                      line=dict(color=RGB_bar[k]),
                                      fillcolor=RGB_bar[k]),row=1, col=2)
    dibujo.update_xaxes(range=[0.5, 1], showgrid=False,title_text="",showticklabels=False, row=1, col=2)
    dibujo.update_yaxes(range=[0, Vref], showgrid=False ,title_text="yaxis 2 title" , row=1, col=2)
    
    return
#------------------------------------------------------------------------------


c_mapa,color_barra = ColorMap()
RGB_bar            = barra_RGB(color_barra)




fig = make_subplots(rows=1, cols=2, column_widths=[0.90, 0.05], subplot_titles=("Plot 1", ""))


Vref = 155
plot_ColorBar(fig,RGB_bar,Vref)
#------------------Crear el colorbar-------------------------------------
#D    = Vref/64
#d    = D/2
#
##fig.add_trace(
##    go.Scatter(x=[2, 3, 4], y=[4, 5, 6], name="yaxis2 data"),
##    secondary_y=True,)
#
#for k in range(64):
#    p_x = 1
#    p_y = k*D+d
#    fig.add_shape(go.layout.Shape(type="rect",x0=p_x-d,y0=p_y-d,x1=p_x+d,y1=p_y+d,
#                                  line=dict(color=RGB_bar[k]),
#                                  fillcolor=RGB_bar[k]),row=1, col=2)
#fig.update_xaxes(range=[0.5, 1], showgrid=False,title_text="xaxis 2 title",showticklabels=False, row=1, col=2)
#fig.update_yaxes(range=[0, Vref], showgrid=False ,title_text="yaxis 2 title" , row=1, col=2)
#------------------------------------------------------------------------------
#fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
#              row=1, col=2)




p_x = 2
p_y = 2
fig.add_shape(go.layout.Shape(type="rect",x0=p_x-0.25,y0=p_y-0.25,x1=p_x+0.25,y1=p_y+0.25,
                              line=dict(color='red'), fillcolor='red'), row=1, col=1 )
fig.update_xaxes(range=[0, 10], showgrid=True,title_text="xaxis 1 title", gridwidth=1, gridcolor='LightPink', row=1, col=1)
fig.update_yaxes(range=[0, 10], showgrid=True,title_text="yaxis 1 title", gridwidth=1, gridcolor='LightPink', row=1, col=1)
#fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)





fig.show()