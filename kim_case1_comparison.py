#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:56:36 2021

@author: Dominik Kern
"""
import numpy as np
import matplotlib.pyplot as plt
import kim_case1_parameters as kc1
model=kc1.KC1()
Nx,  Ny,  dt,  dt_prog,  Nt,  Nci_max,  RelTol_ci=model.get_fem_parameters()

y_mono=np.loadtxt("kim1_y_mono.txt")
p_mono=np.loadtxt("kim1_p_mono.txt")

y_staggered=np.loadtxt("kim1_y_staggered.txt")
p_staggered=np.loadtxt("kim1_p_staggered.txt")

t=0.0
for n in range(Nt):     # time steps
    t += dt
    dt*=dt_prog
    color_code=[0.9*(1-(n+1)/Nt)]*3
    if Nt<21 or n % 10 == 1:
        plt.plot(y_mono, p_mono[n,:],  color=color_code, linestyle='none', marker='o', markersize=6, markerfacecolor='none')    
        plt.plot(y_staggered, p_staggered[n,:],  color=color_code, linestyle='none', marker='x', markersize=6)    
