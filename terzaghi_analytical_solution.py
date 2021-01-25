#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:36:09 2021

@author: Dominik Kern

Analytical solution of 1D Terzaghi problem
"""
import numpy as np
import matplotlib.pyplot as plt
import minimal_model_parameters as mmp

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
    return np.flip(p)   # flip left and right

"""
def terzaghi_example():
    N=50
    L=1
    E=10
    k=0.1
    mu=1
    topload=1

    c=E*k/mu
    x=np.linspace(0.0, L, 10)
    t=np.linspace(1e-4, 0.5, 5)

    for nt in range(len(t)):
        p=dim_sol(x, t[nt], N, L, c, topload)
        plt.plot(x, p)
        
    plt.title('Fourier representation (N='+str(N)+')')
    plt.show()
"""    


model=mmp.MMP()
p_ref, E, nu, k, mu = model.get_physical_parameters()
Nx, Ny, dt, dt_prog, Nt, _, _ = model.get_fem_parameters()
Length, Width, K, Lame1, Lame2, k_mu, cc = model.get_dependent_parameters()
p_ic, p_bc, p_load = model.get_icbc() 

N_Fourier=50
y_ana = np.linspace(0, Length, 101)

y_mono=np.loadtxt("results_y_mono.txt")
p_mono=np.loadtxt("results_p_mono.txt")

y_staggered=np.loadtxt("results_y_staggered.txt")
p_staggered=np.loadtxt("results_p_staggered.txt")

t=0.0
for n in range(Nt):     # time steps
    t += dt
    dt*=dt_prog
    p_ana = dim_sol(y_ana, t, N_Fourier, Length, cc, p_load)
    color_code=[0.9*(1-(n+1)/Nt)]*3
    plt.plot(y_ana, p_ana,  color=color_code)    
    plt.plot(y_mono, p_mono[n,:],  color=color_code, linestyle='none', marker='o', markersize=6, markerfacecolor='none')    
    plt.plot(y_staggered, p_staggered[n,:],  color=color_code, linestyle='none', marker='x', markersize=6)    
