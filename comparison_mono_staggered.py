#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:41:17 2021

@author: d23
"""
import numpy as np
import matplotlib.pyplot as plt

monoN=np.loadtxt("mini_mono_N.txt")
monoT=np.loadtxt("mini_mono_T.txt")

staggeredN=np.loadtxt("mini_staggered_N.txt")
staggeredT=np.loadtxt("mini_staggered_T.txt")

plt.plot(monoN, monoT, 'ro-', staggeredN, staggeredT, 'gx-')
plt.yscale('log')