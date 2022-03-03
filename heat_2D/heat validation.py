# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:34:46 2022

@author: felip
"""
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

# params for domain
height = 1.0
width = 1.0
T2 = 150.0
T1 = 50.0
deltaX = 0.01
deltaY = 0.01
x = np.arange(0, width, deltaX)
y = np.arange(0, height, deltaY)
X, Y = np.meshgrid(x, y)
# X = np.expand_dims(X.flatten(), axis=-1)
# Y = np.expand_dims(Y.flatten(), axis=-1)
n = 1
teta =  ((-1)**(n+1)+1)/n*np.sin(n*np.pi*X/width)*np.sinh(n*np.pi*Y/width)/np.sinh(n*np.pi*height/width)
for n in range(2,100):
    teta += ((-1)**(n+1)+1)/n*np.sin(n*np.pi*X/width)*np.sinh(n*np.pi*Y/width)/np.sinh(n*np.pi*height/width)

teta=2/np.pi*teta
T = teta*(T2-T1)+T1
x = np.arange(-width/2, width/2, deltaX)
y = np.arange(-height/2, height/2, deltaY)
X, Y = np.meshgrid(x, y)

T = T[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(T.min(), T.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('bwr')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(X, Y, T, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title('pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(X[:-1, :-1] + deltaX/2.,
                  Y[:-1, :-1] + deltaY/2., T, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()
