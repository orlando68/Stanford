# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:30:34 2019

@author: 106300
"""

from scipy import signal

system = ([1.0], [1.0, 0.01, 1.0])
t, y = signal.impulse2(system)
t2, y2 = signal.step2(system)
import matplotlib.pyplot as plt
plt.plot(t, y)
plt.plot(t2, y2)