#---------------------------------------------------------------------
# Project : EGNOS TUR Analysis Tool
# Customer : European Space Agency
#---------------------------------------------------------------------
# Author : Enrique Santiago - Deimos Space
# Date : Jan 2019
#---------------------------------------------------------------------
# Copyright Deimos, 2019
# All rights reserved
#---------------------------------------------------------------------
# History: 
#
#---------------------------------------------------------------------
import apps.postprocessing.data as d
import apps.postprocessing.warning_functions as warning
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from apps.postprocessing.navigation_performance.app import app
import pandas as pd


def plot_timeseries(measures, names, title=None, downsample_size=1000, yaxis_name='meters'):
    if d.fewValues:
        downsample_size=None
    
    
    df_aux1 = d.df_solpc.loc[lambda df: df.dataset_number==0]
    t_1 = df_aux1.Time
    u_1 = 1 if downsample_size is None else len(t_1) // downsample_size

    if d.multisession==1:
        df_aux2 = d.df_solpc.loc[lambda df: df.dataset_number==1]
        t_2 = df_aux2.Time
        u_2 = 1 if downsample_size is None else len(t_2) // downsample_size
        t = [t_1,t_2]
        u = [u_1,u_2]
        df_aux = [df_aux1,df_aux2]
        data = [
            go.Scatter(
                x=t[i][::u[i]],
                y=df_aux[i][measure][::u[i]],
                mode='lines',
                name=d.datasets[i]+'-'+name,
                visible= True if measure in ['errorX','Bias','VelX','HPE'] else "legendonly",
            )
            for i in range(len(d.datasets)) for measure, name in zip(measures, names)
        ]
    if d.multisession==0:
        t = [t_1,None]
        u = [u_1,None]
        df_aux = [df_aux1,None]
        i = 0
        data = [
            go.Scatter(
                x=t[i][::u[i]],
                y=df_aux[i][measure][::u[i]],
                mode='lines',
                name=d.datasets[i]+'-'+name
            )
            for measure, name in zip(measures, names)
        ]
    layout = go.Layout(
        title=title,
        xaxis=dict(title='time(s)', exponentformat='none',),
        yaxis=dict(title=yaxis_name, exponentformat='none',),
    )

    return go.Figure(data, layout)


def plot_CEPDRMSR95(downsample_size=1000):
    
    if d.fewValues:
        downsample_size=None
    df_aux1 = d.df_solpc.loc[lambda df: df.dataset_number==0]
    u_1 = 1 if downsample_size is None else len(df_aux1) // downsample_size
    step = .1
    x = np.arange(0, 2 * np.pi + step, step)
    r1, r2 = 2, 4

    if d.multisession==1:
        df_aux2 = d.df_solpc.loc[lambda df: df.dataset_number==1]
        u_2 = 1 if downsample_size is None else len(df_aux2) // downsample_size
        df_aux = [df_aux1,df_aux2]
        u = [u_1,u_2]
        data = [
            go.Scatter(
                x=d.CEP[0] * np.cos(x),
                y=d.CEP[0] * np.sin(x),
                mode='lines',
                name=d.datasets[0]+'-'+'CEP',
            ),
            go.Scatter(
                x=d.drms[0] * np.cos(x),
                y=d.drms[0] * np.sin(x),
                mode='lines',
                name=d.datasets[0]+'-'+'DRMS',
            ),
            go.Scatter(
                x=d.R95[0] * np.cos(x),
                y=d.R95[0] * np.sin(x),
                mode='lines',
                name=d.datasets[0]+'-'+'R95'
            ),
            go.Scatter(
                x=df_aux[0].east[::u[0]],
                y=df_aux[0].down[::u[0]],
                mode='markers',
                name=d.datasets[0]+'-'+'East/Down'
            ),
            go.Scatter(
                x=d.CEP[1] * np.cos(x),
                y=d.CEP[1] * np.sin(x),
                mode='lines',
                name=d.datasets[1]+'-'+'CEP',
                visible= "legendonly", 
            ),
            go.Scatter(
                x=d.drms[1] * np.cos(x),
                y=d.drms[1] * np.sin(x),
                mode='lines',
                name=d.datasets[1]+'-'+'DRMS',
                visible= "legendonly", 
            ),
            go.Scatter(
                x=d.R95[1] * np.cos(x),
                y=d.R95[1] * np.sin(x),
                mode='lines',
                name=d.datasets[1]+'-'+'R95',
                visible= "legendonly", 
            ),
            go.Scatter(
                x=df_aux[1].east[::u[1]],
                y=df_aux[1].down[::u[1]],
                mode='markers',
                name=d.datasets[1]+'-'+'East/Down',
                visible= "legendonly", 
            ),
        ]

    if d.multisession==0:
        
        
        df_aux = [df_aux1,None]
        u = [u_1,None]
        data = [
            go.Scatter(
                x=d.CEP[0] * np.cos(x),
                y=d.CEP[0] * np.sin(x),
                mode='lines',
                name=d.datasets[0]+'-'+'CEP',
            ),
            go.Scatter(
                x=d.drms[0] * np.cos(x),
                y=d.drms[0] * np.sin(x),
                mode='lines',
                name=d.datasets[0]+'-'+'DRMS',
            ),
            go.Scatter(
                x=d.R95[0] * np.cos(x),
                y=d.R95[0] * np.sin(x),
                mode='lines',
                name=d.datasets[0]+'-'+'R95'
            ),
            go.Scatter(
                x=df_aux[0].east[::u[0]],
                y=df_aux[0].down[::u[0]],
                mode='markers',
                name=d.datasets[0]+'-'+'East/Down'
            ),
        ]


    
    if d.multisession:
        boundlimits_axisYmax = np.around(max(d.R95)+0.2,1)
        boundlimits_axisYmin = -np.around(max(d.R95)+0.2,1)
        boundlimits_axisXmax = np.around(max(d.R95)+0.2,1)
        boundlimits_axisXmin = -np.around(max(d.R95)+0.2,1)
    else:
        boundlimits_axisYmax = np.around(max([d.R95[0]])+0.2,1)
        boundlimits_axisYmin = -np.around(max([d.R95[0]])+0.2,1)
        boundlimits_axisXmax = np.around(max([d.R95[0]])+0.2,1)
        boundlimits_axisXmin = -np.around(max([d.R95[0]])+0.2,1) 

    layout = go.Layout(
        title='Accuracy Analysis',
        xaxis=dict(
            title='east',
            range=[boundlimits_axisXmin, boundlimits_axisXmax]
        ),
        yaxis=dict(
            title='north',
            range=[boundlimits_axisYmin, boundlimits_axisYmax]
        )
    )

    return go.Figure(data, layout)


def dataframe_to_table(df, format=None, layout=None, col_names={'Time': 'time(s)'}):
    trace = go.Table(
        header=dict(
            # values=df.columns,
            values=list(map(lambda x: col_names[x] if x in col_names else x, df.columns)),
            fill=dict(color='#C2D4FF'),
        ),
        cells=dict(
            values=[
                df[column] for column in df.columns
            ],
            fill=dict(color='#F5F8FF'),
            format=format,
        )
    )
    data = [trace]
    if layout is None:
        layout = dict(
            margin=dict(
                t=40,
                r=40,
                l=40,
                b=40,
            ),
        )
    return go.Figure(data, layout)

def statistical_table():
    if d.multisession==1:
        df_aux1 = d.accuracy_parameters_table[0].assign(dataset = d.datasets[0])
        df_aux2 = d.accuracy_parameters_table[1].assign(dataset = d.datasets[1])
        df_aux = pd.concat([df_aux1,df_aux2])
        df_aux = df_aux.loc[slice(None),['parameter','dataset', 'value', 'occurrences under value', 'under %', ]]

    if d.multisession==0:
        df_aux = d.accuracy_parameters_table[0].assign(dataset = d.datasets[0])
        df_aux = df_aux.loc[slice(None),['parameter','dataset', 'value', 'occurrences under value', 'under %', ]]

    fig = dataframe_to_table(df_aux,format=[None]*2 + ['.4f'] + [None] * 2)

    return fig

def get_layout():
    
    ##########################################################################
    # def get_layout()
    # This function gives the layout to the Dash Window. Dropdowns, user controls,
    # plots and tables are loaded on it.
    # When fixed receiver is selected and multisession mode is available,
    # a reduced set of time series are shown in the plots. This is made in order to
    # improve the visualization of the data. The user can select all available data
    # clicking on the plot legend.  
    # returns: Dash web application layout. 
    # Author : Enrique Santiago
    # version: 2.01
    # Date : 1 Mar, 2019
    #-------------------------------------------------------------------------
    # Copyright Deimos, 2018
    # All rights reserved
    #-------------------------------------------------------------------------
    # History: 
    # 
    #-------------------------------------------------------------------------

    if d.df_solpc.empty:
        
        return warning.notAvailableCapability('Not Available data',
            'It is necessary to provide solpc file (PVT data) in order to compute accuracy metrics')
     
    layout = html.Div([
        html.Div([
            html.Div([
                dcc.Graph(
                    id='HPE',
                    figure=plot_timeseries(['HPE', 'VPE'], ['HPE', 'VPE'], title='Horizontal & Vertical Position Error'),
                ),
            ], className='col s6'),
            html.Div([
                dcc.Graph(
                    id='CEPDRMSR95',
                    figure=plot_CEPDRMSR95(),
                ),
            ], className='col s6'),
        ], className='row'),
        dcc.Graph(
            id='accuracy-parameters-table',
            figure=statistical_table(),
        ),
    ])
    return layout


name = 'Standford Analysis'
layout_function = get_layout
path = '/postprocessing/navigation-performance/np3'

app.add_app(name, layout_function, path)


