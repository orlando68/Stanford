#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:21:20 2019

@author: instalador
"""



import plotly.graph_objects as go

trace1 = go.Scatter(
    x=[1, 2, 3],
    y=[4, 5, 6]
)
trace2 = go.Scatter(
    x=[20, 30, 40],
    y=[50, 60, 70],
    xaxis="x2",
    yaxis="y2"
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis1=dict(
        domain=[0, 0.7]
    ),
    xaxis2=dict(
        domain=[0.8, 1]
    ),
#    yaxis2=dict(
#        anchor="x2"
#    )
)
fig = go.Figure(data=data, layout=layout)
fig.show()

fig = make_subplots(rows=1, cols=2, column_widths=[2, 0.05], subplot_titles=("Plot 1", ""),specs=[[{"secondary_y": False}, {"secondary_y": True}]] )
