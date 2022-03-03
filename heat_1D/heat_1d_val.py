#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 00:01:20 2022

@author: felipeeler
"""

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.special import erf

# params for domain
L = float(0.10)                  # m
Ti = float(100.0)               # C
Ts = float(25.0)                # C
tf = 100.0		# s

# physical parameters
alpha = float(15.0e-6)         # m2/s

# make validation data
deltaTT = 10
deltaX = 0.001
x = np.arange(0, L, deltaX)
t = np.arange(0, tf, deltaTT)
X, TT = np.meshgrid(x, t)
# X = np.expand_dims(X.flatten(), axis=-1)
# TT = np.expand_dims(TT.flatten(), axis=-1)
   
# Analytical Solution using 25 terms of Eq. 5.42
T = erf(X/(2*(alpha*TT)**0.5))*(Ti-Ts)+Ts
T = np.nan_to_num(T,nan=Ts)

T = T[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(T.min(), T.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('jet')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(X, TT, T, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title('pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(X[:-1, :-1] + deltaX/2.,
                  TT[:-1, :-1] + deltaTT/2., T, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()
