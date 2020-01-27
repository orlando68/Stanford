# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 10:06:29 2020

@author: 106300
"""

# import plotly.graph_objects as go
import plotly.graph_objs as go
fig = go.Figure()



a = 10
print(a)
# Create scatter trace of text labels
fig.add_trace(go.Scatter(
    x=[2, 3.5, 6],
    y=[1, 1.5, 1],
    text=["Vertical Line",
          "Horizontal Dashed Line",
          "Diagonal dotted Line"],
    mode="text",
))

# Set axes ranges
fig.update_xaxes(range=[0, 7])
fig.update_yaxes(range=[0, 2.5])

# Add shapes
fig.add_shape(
        # Line Vertical
        go.layout.Shape(
            type="line",
            x0=1,
            y0=0,
            x1=1,
            y1=2,
            line=dict(
                color="RoyalBlue",
                width=3
            )
))
fig.add_shape(
        # Line Horizontal
        go.layout.Shape(
            type="line",
            x0=2,
            y0=2,
            x1=5,
            y1=2,
            line=dict(
                color="LightSeaGreen",
                width=4,
                dash="dashdot",
            ),
    ))
fig.add_shape(
        # Line Diagonal
        go.layout.Shape(
            type="line",
            x0=4,
            y0=0,
            x1=6,
            y1=2,
            line=dict(
                color="MediumPurple",
                width=4,
                dash="dot",
            )
))
fig.update_shapes(dict(xref='x', yref='y'))
fig.show()