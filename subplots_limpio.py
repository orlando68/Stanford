#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:03:23 2019

@author: instalador
"""
import numpy as np

import plotly.graph_objects as go

from plotly.subplots import make_subplots

#from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
#from collections import OrderedDict


import dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)




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
def plot_ColorBar(dib,RGB_bar,Vref):
    D    = Vref/64
    d    = D/2
    
    for k in range(64):
        p_x = 1
        p_y = k*D+d
        dib.add_shape(go.layout.Shape(type="rect",x0=p_x-d,y0=p_y-d,x1=p_x+d,y1=p_y+d,
                                      line=dict(color=RGB_bar[k]),
                                      fillcolor=RGB_bar[k]),secondary_y=True,row=1, col=2)
    dib.update_xaxes(range=[0.5, 1], showgrid=False,title_text="",showticklabels=False, row=1, col=2)
#    dib.update_yaxes(range=[0, Vref], showgrid=False ,title_text="yaxis izda2 title" ,secondary_y=False, row=1, col=2)
    dib.update_yaxes(range=[0, Vref], showgrid=False ,title_text="yaxis dcha2 title" ,secondary_y=True, row=1, col=2)
    
    return
#------------------------------------------------------------------------------
def add_text(dib,texto,x0,y0,color):
    dib.add_trace(go.Scatter(x=x0,y=y0,text=texto, mode="text",textposition="middle center",
#                             textfont=dict(family="sans serif", size=18, color=color )
                             ), row=1, col=1)
    return
#------------------------------------------------------------------------------
def add_patch(dib,p_x,p_y, color):
    dib.add_shape(go.layout.Shape(type="rect",x0=p_x-0.25,y0=p_y-0.25,x1=p_x+0.25,y1=p_y+0.25,
                              line=dict(color=color), fillcolor=color),layer = "below", row=1, col=1 )
    return
#------------------------------------------------------------------------------
def add_polygon(dib,x_points,y_points,color):
    camino = 'M' 
    for index,k in enumerate(x_points):
        camino = camino + ' '+ str(x_points[index]) +','+str(y_points[index])
        if index < np.size(x_points)-1:
            camino = camino + ' L'
    camino = camino + ' Z'
    print(camino)
    
    dib.add_shape(go.layout.Shape(type="path",path=camino,fillcolor=color),line_color=color,layer = "below", opacity=1.0,row=1, col=1)
    
    
    return
#------------------------------------------------------------------------------
    
c_mapa,color_barra = ColorMap()
RGB_bar            = barra_RGB(color_barra) # to generate in  rgb(0.267004,0.004874,0.329415)




fig = make_subplots(rows=1, cols=2, column_widths=[10, 0.5], subplot_titles=("Plot 1", r'$HPL_{pepe} [m]$'),specs=[[{"secondary_y": False}, {"secondary_y": True}]])



Vref = 155
plot_ColorBar(fig,RGB_bar,Vref)


p_x = 2.3
p_y = 2.1

add_polygon(fig,[4,4,3,4],[1,5,3,4],'rgb(255,114,111)')
add_polygon(fig,[7,2,3,4],[3,4,3,4],'rgb(255,204,203)')
add_polygon(fig,[8,2,6,4],[9,11,11,4],'rgb(255,158,87)')
add_polygon(fig,[4,2,9,3],[20,10,3,4],'rgb(253,255,143)')
add_polygon(fig,[6,9,3,9],[30,20,9,4],'pink')
add_patch  (fig,p_x,p_y,'red')
#add_text   (fig,'orlando',[5],[5],'black')
add_text   (fig,['pepe','potamo'],[5,5],[7,9],'black')

fig.update_xaxes(range=[0, 10], showgrid=True,title_text="<b>secondary</b> yaxis title xaxis 1 title", gridwidth=1, gridcolor='LightPink',layer = "below traces", row=1, col=1)
fig.update_yaxes(range=[0, 10], showgrid=True,title_text="yaxis 1 title", gridwidth=10, gridcolor='LightPink', layer = "above traces",row=1, col=1)

fig.show()