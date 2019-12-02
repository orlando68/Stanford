#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 17:30:51 2019

@author: instalador
"""

import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
                    z=[["rgb(60, 20, 0)", 1, "rgb(0, 0, 0)" ],
                      ],
                        colorbar=dict(
        tick0=0,
        dtick=5)))
fig.show()