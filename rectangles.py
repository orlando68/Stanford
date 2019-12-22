# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:32:32 2019

@author: 106300
"""


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1.5, 4.5],
    y=[0.75, 0.75],
    text=["Unfilled Rectangle", "Filled Rectangle"],
    mode="text",
))

# Set axes properties
fig.update_xaxes(range=[0, 7], showgrid=False)
fig.update_yaxes(range=[0, 3.5])

# Add shapes
fig.add_shape(
        # unfilled Rectangle
        go.layout.Shape(
            type="rect",
            x0=[1,2],
            y0=[1,2],
            x1=[2,3],
            y1=[3,4],
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
app.run_server(debug=True)