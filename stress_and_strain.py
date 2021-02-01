#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:48:53 2021

@author: d23
"""
# TODO detect dimension 2 or 3

import fenics as fe

# Define strain and stress (2D plain strain)
class SAS:
    
    def __init__(self, Lame1, Lame2, K):
        self.Lame1=Lame1
        self.Lame2=Lame2
        self.K=K

    def epsilon(self, u):
        return fe.sym(fe.grad(u))

    def sigma_eff(self, u):
        return self.Lame1*fe.div(u)*fe.Identity(2) + 2*self.Lame2*self.epsilon(u)

    def sigma(self, p, u):
        return self.sigma_eff(u) - p * fe.Identity(2)

    def sv(self, p, u):
        return self.K*fe.div(u) - p

