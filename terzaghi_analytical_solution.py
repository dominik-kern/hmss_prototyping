#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:36:09 2021

@author: Dominik Kern

Analytical solution of 1D Terzaghi problem
"""
import numpy as np
import matplotlib.pyplot as plt


def nondim_sol(x, t, N):
    P=0*x
    for n in np.arange(N):
        kappa=(0.5+n)*np.pi
        S=2/kappa
        T=np.exp(-t*kappa**2)
        X=np.sin(kappa*x)
        P=P+S*T*X
    return P

def dim_sol(x, t, N, L, c, topload):
    tau=(L**2)/c
    X=x/L
    T=t/tau
    p=nondim_sol(X, T, N)/topload
    return p


N=50

L=2
E=5
k=0.1
mu=0.3
topload=4.0

c=E*k/mu

x=np.linspace(0.0, L, 100)
t=np.linspace(1e-4, 0.5, 10)

for nt in range(len(t)):
    p=dim_sol(x, t[nt], N, L, c, topload)
    plt.plot(x, p)
    
plt.show()

#outfile = "test.npy"
#x = np.arange(10)
#np.save(outfile, x)
#np.load(outfile)


