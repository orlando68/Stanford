# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:24:47 2019

@author: 106300
"""

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1.5, 4.5],
    y=[0.75, 0.75],
    text=["Unfilled Rectangle", "Filled Rectangle"],
    mode="text",
))

# Set axes properties
fig.update_xaxes(range=[0, 7], showgrid=False)
fig.update_yaxes(range=[0, 5])

# Add shapes
fig.add_shape(
        # unfilled Rectangle
        go.layout.Shape(
            type="rect",
            xref="x",
            yref="y",
            x0=1,
            y0=1,
            x1=2,
            y1=2,
            line=dict(
                color="RoyalBlue",
            ),
        ))
fig.add_shape(
        # filled Rectangle
        go.layout.Shape(
            type="rect",
            x0=3,
            y0=1,
            x1=6,
            y1=2,
            line=dict(
                color="RoyalBlue",
                width=2,
            ),
            fillcolor="LightSkyBlue",
        ))
fig.update_shapes(dict(xref='x', yref='y'))
fig.show()