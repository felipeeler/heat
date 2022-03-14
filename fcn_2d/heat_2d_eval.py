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


# Predicted Data

temp1 = np.load('./network_checkpoint_heat_2d/val_domain/results/Val_pred.npz', allow_pickle=True)
temp2 = temp1['arr_0']
data_pred = temp2.item(0)
x_pred = data_pred['x']
y_pred = data_pred['y']
T_pred = data_pred['T']

X_pred = np.reshape(x_pred, (100,100))
Y_pred = np.reshape(y_pred, (100,100))
T_pred = np.reshape(T_pred, (100,100))


levels = MaxNLocator(nbins=15).tick_values(T_pred.min(), T_pred.max())

cmap = plt.get_cmap('jet')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)


fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
fig.set_size_inches(12, 3.5)

pred = ax0.contourf(X_pred,
                  Y_pred, T_pred, levels=levels,
                  cmap=cmap)
fig.colorbar(pred, ax=ax0, label='Temperature (°C)')
ax0.set_title('Modulus')
ax0.set_xlabel('Lenght (m)')
ax0.set_ylabel('Height (m)')

# real data

temp3 = np.load('./network_checkpoint_heat_2d/val_domain/results/Val_true.npz', allow_pickle=True)
temp4 = temp3['arr_0']
data_true = temp4.item(0)
x_true = data_true['x']
t_true = data_true['y']
T_true = data_true['T']

X_true = np.reshape(x_true, (100,100))
Y_true = np.reshape(t_true, (100,100))
T_true = np.reshape(T_true, (100,100))


levels1 = MaxNLocator(nbins=15).tick_values(T_true.min(), T_true.max())



real = ax1.contourf(X_true,
                  Y_true, T_true, levels=levels1,
                  cmap=cmap)
fig.colorbar(real, ax=ax1, label='Temperature (°C)')
ax1.set_title('Analytical Solution')
ax1.set_xlabel('Lenght (m)')
ax1.set_ylabel('Height (m)')

# Difference

temp5 = np.load('./network_checkpoint_heat_2d/val_domain/results/Val_diff.npz', allow_pickle=True)
temp6 = temp5['arr_0']
data_diff = temp6.item(0)
x_diff = data_diff['x']
t_diff = data_diff['y']
T_diff = data_diff['T']

X_diff = np.reshape(x_diff, (100,100))
Y_diff = np.reshape(t_diff, (100,100))
T_diff = np.reshape(T_diff, (100,100))


levels2 = MaxNLocator(nbins=15).tick_values(T_diff.min(), T_diff.max())


diff = ax2.contourf(X_diff, Y_diff, T_diff, levels=levels2, cmap=cmap)
fig.colorbar(diff, ax=ax2, label='Temperature (°C)')
ax2.set_title('Difference')
ax2.set_xlabel('Lenght (m)')
ax2.set_ylabel('Height (m)')



# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

fig.savefig('FCN_6x64.png', dpi=300)
