# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:00:45 2019

@author: 106300
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

p_x = 1
p_y = 1
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
#    html.H1(
#        children='Hello Dash',
#        style={
#            'textAlign': 'center',
#            'color': colors['text']
#        }
#    ),

    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='example-graph-2',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    ),

    dcc.Graph(
        id='example-graph-3',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [1, 1, 1], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [1, 1, 1], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    ),
    dcc.Graph(
        id='example-graph-4',
    go.layout.Shape(type="rect",x0=p_x-0.25,y0=p_y-0.25,x1=p_x+0.25,y1=p_y+0.25,line=dict(color='red'), fillcolor='red')
    )
    
    
])

if __name__ == '__main__':
    app.run_server(debug=True)