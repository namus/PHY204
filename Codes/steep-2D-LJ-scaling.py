#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:52:04 2019

@author: Suman
"""

import matplotlib.pyplot as plt
import numpy as np

#############################################################
def force(pos):
    f = np.zeros((len(pos),2))
    for i in range(len(pos)-1):
        for j in range(i+1, len(pos)):
            r = np.linalg.norm(pos[i]-pos[j])
            fij = (6/r**2)*(2/r**12 - 1/r**6)*(pos[i]-pos[j])
            f[i] += fij 
            f[j] -= fij 
    # Compute maximum force (magnitude)
    maxf = np.linalg.norm(f,axis=1).max() 
    return maxf, f # Return a tuple: both max f, and force array
###############################################################
def potnrg(pos):
    nrg = 0.0
    for i in range(len(pos)-1):
        for j in range(i+1, len(pos)):
            r = np.linalg.norm(pos[i]-pos[j])
            nrg += 1/r**12 - 1/r**6
    return nrg
############################################################
# Initialise coordinates on a square grid with spacing 1.5 units
def create_grid(npart):
    pos = np.zeros((npart,2))
    for i in range(boxl):
        for j in range(boxl):
            pos[boxl*i+j] = 0.9*np.array([i,j])
    return pos
#########################################################
def create_random(npart, spacing):
    return np.random.random((npart,2))*spacing 
########################################################

#Initialise ...
boxl = 5
npart = boxl**2 # Number of particles
steps = 1000
tol = 1.0e-6
###########################################################

pos = create_random(npart, 1.3)
#pos = create_grid(npart)
pos_init = pos.copy() # Initial position saved for plotting later

#Steepest descent starts
for step in range(steps):
    alpha = 0.01
    maxf, f = force(pos)
    if maxf > 10:
        alpha /= maxf
    pos += alpha*f

    print(step, ": Energy = ", potnrg(pos), "Max. Force = ", maxf)

    if maxf < tol: # Convergence is checked by max force value, not energy
        #np.savetxt("LJ-2D-config.dat", pos) # Uncomment if you want to save final coord
        print(f"Converged after {step} steps!")
        break
    
# Compare initial and final coordinates
plt.figure(1,figsize=(18, 6))
# Initial coordinates
plt.subplot(121)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Coordinates: Before')
plt.scatter(pos_init[:,0],pos_init[:,1],s=100,c="b") # Before

# Final coordinates
plt.subplot(122)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Coordinates: After')
plt.scatter(pos[:,0],pos[:,1],s=100,c="r") # After
plt.show()














