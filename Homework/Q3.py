#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:19:18 2020

@author: Suman
"""

import numpy as np

x = 0.0 # Initial value of x
x4av = 0.0
x4avold = 100 

maxdx = 0.5
totstep = 0

while totstep <= 1000000:
    totstep += 1
    xtry = x + np.random.uniform(-maxdx, maxdx)
    dE = xtry**4 - x**4 # Analogous to dE in MC simulation in 1D SHO!
    if (np.random.random() <= np.exp(-dE)):
        x = xtry

    x4av += x**4 # Sampling the average
       
    if (totstep % 1000 == 0):
        Error = abs((x4avold - x4av)/totstep)
        print(f"Step: {totstep}, Mean: {x4av/totstep}, Error: {Error}")
        x4avold = x4av # Updating the reference for convergence checking
        if (abs(Error) < 1.0e-3):
            print("Converged!")
            break
        
print("Thank you! :)")
