#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:50:43 2019

@author: instalador
"""

#import plotly.express as px
#iris = px.data.iris()
#fig = px.scatter(iris, x="sepal_width", y="sepal_length",
#                 color="sepal_length")#, color_continuous_scale=px.colors.sequential.Viridis)
#
#fig.show()

import plotly.express as px
iris = px.data.iris()
fig = px.density_heatmap(iris, x="sepal_width", y="sepal_length")
fig.show()