#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
"""
Created on Fri Jan 22 16:32:08 2021

Parameters and functions for Kim case 1 (Terzaghi)

"""

class KC1:
    def __init__(self):
    # physical parameters
        self.p_ref = 2.125e6    # reference pressure
        self.E = 3.0*100e6  # Young's modulus (bulk, drained)
        self.nu = 0.0;     # Poisson ratio (bulk)
        self.k = 50.0e-15   # permeability
        self.mu = 1.0e-3    # viscosity
        self.rho_f = 1.0e+3 # fluid density 
        self.tau = 0.83 # 1.21 coupling
        self.beta_s = 0 # solid compressibility
        self.phi = 0.3 # porosity

    # FEM parameters    
        self.Nx=1        # mesh divisions x-direction (2)
        self.Ny=5       # mesh divisions y-direction  (15)
        self.dt=400.0 # initial time step
        self.dt_prog=1.0 # time step progression
        self.Nt=5   # number of time steps
        
    # dependent parameters
        self.Length = 30 # Length (y)
        self.Width = 1 # Width (x)
        self.Lame1 = self.E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.Lame2 = self.E/(2*(1+self.nu)) 
        self.K=self.Lame1 + (2.0/3.0)*self.Lame2  
        self.k_mu = self.k/self.mu
        self.cc = self.E*self.k_mu # consolidation coefficient TODO with storage
        self.alpha = 1 - self.beta_s * self.K  # Biot coefficient
        self.beta_f = ((self.alpha**2)/(self.tau*self.K)-(self.alpha-self.phi)*self.beta_s)/(self.phi)
        self.S = self.phi*self.beta_f + (self.alpha-self.phi)*self.beta_s
        self.tc_DK = (self.Length**2)/self.cc    # characteristic time (Dominik Kern)
        mv=1.0/(self.K+(4.0/3.0)*self.Lame2)   # P-wave modulus
        ccTD=self.k/( self.mu*(mv*self.alpha**2 + self.S) )
        self.tc_TD = (self.Length**2)/ccTD  # characteristic time (Tengfei Deng)
        
        # only for staggered
        self.Nci_max=100   # maximal number of coupling iterations
        self.RelTol_ci=1.0e-10   # relative tolerance of coupling iterations
        self.betaFS=0.5*(self.alpha**2)/self.K  # optimal 0.7 * 0.5*(alpha**2)/K1.0   
    
    # initial and boundary conditions
        self.p_ic=1*self.p_ref
        self.p_bc=1*self.p_ref    # H BC
        self.p_load=2*self.p_ref  # HM BC

        
    def get_physical_parameters(self):
        return [self.p_ref, self.E, self.nu, self.k, self.mu, self.rho_f, self.tau, self.beta_s, self.phi]

    def get_fem_parameters(self):
    # staggered and mono
        return [self.Nx, self.Ny, self.dt, self.dt_prog, self.Nt, self.Nci_max, self.RelTol_ci, self.betaFS]

    def get_dependent_parameters(self):
        return [self.Length, self.Width, self.K, self.Lame1, self.Lame2, self.k_mu, self.cc, self.alpha, self.S, self.tc_DK, self.tc_TD]

    def get_icbc(self):
        return [self.p_ic, self.p_bc, self.p_load]
