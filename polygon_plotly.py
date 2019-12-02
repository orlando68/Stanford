#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:40:52 2019

@author: instalador
"""

#def PolygonSort(corners):
#    n = len(corners)
#    cx = float(sum(x for x, y in corners)) / n
#    cy = float(sum(y for x, y in corners)) / n
#    cornersWithAngles = []
#    for x, y in corners:
#        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
#        cornersWithAngles.append((x, y, an))
#    cornersWithAngles.sort(key = lambda tup: tup[2])
#    return map(lambda (x, y, an): (x, y), cornersWithAngles)
#
#def PolygonArea(corners):
#    n = len(corners)
#    area = 0.0
#    for i in range(n):
#        j = (i + 1) % n
#        area += corners[i][0] * corners[j][1]
#        area -= corners[j][0] * corners[i][1]
#    area = abs(area) / 2.0
#    return area

corners = [(0, 0), (3, 0), (2, 10), (3, 4), (1, 5.5)]
corners_sorted = corners#PolygonSort(corners)
area = 10

x = [corner[0] for corner in corners_sorted]
y = [corner[1] for corner in corners_sorted]

annotation = go.Annotation(
    x=5.5,
    y=8.0,
    text='The area of the polygon is approximately %s' % (area),
    showarrow=False
)

trace1 = go.Scatter(
    x=x,
    y=y,
    mode='markers',
    fill='tozeroy',
)

layout = go.Layout(
    annotations=[annotation],
    xaxis=dict(
        range=[-1, 9]
    ),
    yaxis=dict(
        range=[-1, 12]
    )
)

trace_data = [trace1]
fig = go.Figure(data=trace_data, layout=layout)

py.iplot(fig, filename='polygon-area')

