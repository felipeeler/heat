#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 00:01:20 2022

@author: felipeeler
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


temp1 = np.load('./network_checkpoint_heat/val_domain/results/Val_pred.npz', allow_pickle=True)
temp2 = temp1['arr_0']
data_pred = temp2.item(0)
x_pred = data_pred['x']
t_pred = data_pred['t']
T_pred = data_pred['T']

X_pred = np.reshape(x_pred, (200,200))
Y_pred = np.reshape(t_pred, (200,200))
T_pred = np.reshape(T_pred, (200,200))


T_pred = T_pred
levels = MaxNLocator(nbins=15).tick_values(T_pred.min(), T_pred.max())


# Predicted Data
cmap = plt.get_cmap('jet')

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
fig.set_size_inches(12, 8)

im = ax0.contourf(X_pred,
                  Y_pred, T_pred, levels=levels,
                  cmap=cmap)
fig.colorbar(im, ax=ax0,label='Temperature (°C)')
ax0.set_title('Modulus')
ax0.set_xlabel('Lenght (m)')
ax0.set_ylabel('time (s)')


# real data

temp3 = np.load('./network_checkpoint_heat/val_domain/results/Val_true.npz', allow_pickle=True)
temp4 = temp3['arr_0']
data_true = temp4.item(0)
x_true = data_true['x']
t_true = data_true['t']
T_true = data_true['T']

X_true = np.reshape(x_true, (200,200))
Y_true = np.reshape(t_true, (200,200))
T_true = np.reshape(T_true, (200,200))


T_true = T_true
levels1 = MaxNLocator(nbins=15).tick_values(T_true.min(), T_true.max())



cf = ax1.contourf(X_true,
                  Y_true, T_true, levels=levels1,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1,label='Temperature (°C)')
ax1.set_title('Analytical Solution')
ax1.set_xlabel('Lenght (m)')
ax1.set_ylabel('time (s)')

# Difference

temp5 = np.load('./network_checkpoint_heat/val_domain/results/Val_diff.npz', allow_pickle=True)
temp6 = temp5['arr_0']
data_diff = temp6.item(0)
x_diff = data_diff['x']
t_diff = data_diff['t']
T_diff = data_diff['T']

X_diff = np.reshape(x_diff, (200,200))
Y_diff = np.reshape(t_diff, (200,200))
T_diff = np.reshape(T_diff, (200,200))


T_diff = T_diff
levels2 = MaxNLocator(nbins=15).tick_values(T_diff.min(), T_diff.max())


cf = ax2.contourf(X_diff,
                  Y_diff, T_diff, 
                  levels=levels2,
                  cmap=cmap)
fig.colorbar(cf, ax=ax2,label='Temperature (°C)')
ax2.set_title('Difference')
ax2.set_xlabel('Lenght (m)')
ax2.set_ylabel('time (s)')


# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()
fig.savefig('FCN_3x256.png', dpi=300)


