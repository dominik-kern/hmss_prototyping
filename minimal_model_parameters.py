#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
Created on Fri Jan 22 16:32:08 2021

Parameters and functions for minimal model

"""

class MMP:
    def __init__(self):
    # physical parameters
        self.p_ref = 2.0    # reference pressure
        self.E = 10.0  # Young's modulus (bulk, drained)
        self.nu = 0.0;     # Poisson ratio (bulk)
        self.k = 0.1   # permeability
        self.mu = 1.0    # viscosity

    # FEM parameters    
        self.Nx=20        # mesh divisions x-direction
        self.Ny=20       # mesh divisions y-direction
        self.dt=0.01 # initial time step
        self.dt_prog=1.0 # time step progression
        self.Nt=10   # number of time steps
    # only for staggered
        self.Nci_max=100   # maximal number of coupling iterations
        self.RelTol_ci=1.0e-10   # relative tolerance of coupling iterations
        
    # initial and boundary conditions
        self.p_ic=1*self.p_ref
        self.p_bc=1*self.p_ref    # H BC
        self.p_load=3*self.p_ref  # HM BC

    # dependent parameters
        self.Length = 1 # unit square!
        self.Width = 1 # unit square!
        self.Lame1 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.Lame2 = self.E/(2*(1+self.nu)) 
        self.K=self.Lame1 + (2.0/3.0)*self.Lame2  
        self.k_mu = self.k/self.mu
        self.cc = self.E*self.k_mu # consolidation coefficient
        
        
    def get_physical_parameters(self):
        return [self.p_ref, self.E, self.nu, self.k, self.mu]

    def get_fem_parameters(self):
    # staggered and mono
        return [self.Nx, self.Ny, self.dt, self.dt_prog, self.Nt, self.Nci_max, self.RelTol_ci]

    def get_dependent_parameters(self):
        return [self.Length, self.Width, self.K, self.Lame1, self.Lame2, self.k_mu, self.cc]

    def get_icbc(self):
        return [self.p_ic, self.p_bc, self.p_load]
