# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:45:59 2019

@author: 106300
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

from plotly.subplots import make_subplots
print('-------------------------------------')
p_x= 1
p_y = 2

fig = make_subplots(rows=1, cols=2, column_widths=[10, 0.5], subplot_titles=("Plot 1", r'$HPL_{pepe} [m]$'),specs=[[{"secondary_y": False}, {"secondary_y": True}]])



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app                  = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout           = html.Div(children=[ html.H1(children='Hello Dash'),
                                          html.Div(children='''Dash: A web application framework for Python. Al loro!! '''),

                                          dcc.Graph(id='example-graphiii', figure={ 'data': [{'x': [1, 2, 3], 'y': [4, 10, 2], 'type': 'bar', 'name': 'SF'},
                                                                                             {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'} ],
                                                                                    'layout': {  'title': 'Dash Data Visualization'  }}  ),
                                fig.add_shape(go.layout.Shape(type="rect",x0=p_x-0.25,y0=p_y-0.25,x1=p_x+0.25,y1=p_y+0.25,
                              line=dict(color='red'), fillcolor='blue'),layer = "below", row=1, col=1 )
                                         ]
                                )

if __name__ == '__main__':
    app.run_server(debug=True)

