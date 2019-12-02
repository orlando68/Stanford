# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:45:59 2019

@author: 106300
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app                  = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app.layout           = html.Div(children=[ html.H1(children='Hello Dash'),
#
#    html.Div(children='''Dash: A web application framework for Python. Al loro!! '''),
#
#    dcc.Graph(id='example-graphiii', figure={ 'data': [{'x': [1, 2, 3], 'y': [4, 10, 2], 'type': 'bar', 'name': 'SF'},
#                          {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'} ],
#                 'layout': {  'title': 'Dash Data Visualization'  }}
#    )
#])

#if __name__ == '__main__':
#    app.run_server(debug=True)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app                  = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout           = html.Div(children=[html.H1(children='Hello Dash'), 
                                          html.H1(children='segundo nivel'),
                                          html.Div(children='''UNO'''),
                                          html.Div(children=html.H1(children='segundo nivel')),
                                          html.Div(children= [  html.H1(children='Hello Dash'), html.H1(children='segundo nivel') ]),
                                          ]
                                )

if __name__ == '__main__':
    app.run_server(debug=True)