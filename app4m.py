# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:22:18 2019

@author: 106300
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go


def add_patch(dib,p_x,p_y, color):
    dib.add_shape(go.layout.Shape(type="rect",x0=p_x-0.25,y0=p_y-0.25,x1=p_x+0.25,y1=p_y+0.25,
                              line=dict(color=color), fillcolor=color),layer = "below", row=1, col=1 )
    return
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/' +
    '5d1ea79569ed194d432e56108a04d188/raw/' +
    'a9f9e8076b837d541398e999dcbac2b2826a81f8/'+
    'gdp-life-exp-2007.csv')

p_x = 1
p_y = 1
fig = go.Figure()
app.layout = html.Div([
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

])
print('primero paso por aqui')

if __name__ == '__main__':
    print(df)
    app.run_server(debug=True)