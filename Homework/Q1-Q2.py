#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:12:11 2020

@author: Suman
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def potnrg(config, h, lenx):
    for x in range(lenx):
        for y in range(lenx):
            for z in range(lenx):
                nrg  = config[np.mod(x-1, lenx), y, z] * config[x,y,z]
                nrg += config[np.mod(x+1, lenx), y, z] * config[x,y,z]
                nrg += config[x, np.mod(y-1, lenx), z] * config[x,y,z]
                nrg += config[x, np.mod(y+1, lenx), z] * config[x,y,z]
                nrg += config[x, y, np.mod(z-1, lenx)] * config[x,y,z]
                nrg += config[x, y, np.mod(z+1, lenx)] * config[x,y,z]
            
    return -0.5*nrg - h*config.sum()

@jit(nopython=True)
def delnrg(config, x, y, z, h, lenx):
    dE = 0.0    
    dE += config[np.mod(x-1, lenx), y, z]
    dE += config[np.mod(x+1, lenx), y, z]
    dE += config[x, np.mod(y-1, lenx), z]
    dE += config[x, np.mod(y+1, lenx), z]
    dE += config[x, y, np.mod(z-1, lenx)]
    dE += config[x, y, np.mod(z+1, lenx)]
    
    return 2.0*config[x,y,z]*dE + 2.0*h*config[x,y,z]

@jit(nopython=True)
def run_MC(lenx, kT, h, steps, equib):
    config = np.ones((lenx,lenx,lenx)) # For all up initial config
    #config = np.random.choice([-1,1], size=(lenx,lenx)) # for random config
    nspins = lenx*lenx*lenx
    mag = config.mean()
    nrg = potnrg(config, h, lenx)
    acc = 0
    mfull = np.zeros(steps-equib)
    nrgfull = np.zeros(steps-equib)

    for step in range(steps):
        for _ in range(nspins):
            x = np.random.randint(lenx)
            y = np.random.randint(lenx)
            z = np.random.randint(lenx)
            
            dE = delnrg(config, x, y, z, h, lenx)
                    
            if (np.random.random() < np.exp(-dE/kT)):
                config[x, y, z] *= -1
                nrg += dE
                acc += 1
            
        mag = config.mean()

        if step >= equib:
            mfull[step-equib] = mag
            # Energy per particle
            nrgfull[step-equib] = nrg / (lenx*lenx*lenx) 

    return mfull, nrgfull

# Main driver code for temperature variation starts here ...

lenx = 10
h = 0.0
steps = 200000 
equib = 10000 
mfull = np.zeros(steps-equib)
nrgfull = np.zeros(steps-equib)

listkT = np.arange(3.5,5.5,0.05)

Cv = np.zeros(len(listkT))
avnrg = np.zeros(len(listkT))
avmag = np.zeros(len(listkT))
U4 = np.zeros(len(listkT))

for indx, kT in enumerate(listkT):
    mfull, nrgfull = run_MC(lenx, kT, h, steps, equib)
    print(f"Finished kT = {kT:0.2f} ...")
    avnrg[indx] = np.mean(nrgfull)
    Cv[indx] = np.var(nrgfull)/(kT*kT)
    avmag[indx] = np.mean(mfull)
    U4[indx] = 1.0 - np.mean(mfull**4)/(3*np.mean(mfull**2)**2)

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8,8))

ax1.set_ylabel("<E>")
ax2.set_ylabel("$C_{v}$")
ax3.set_ylabel("<m>")
ax4.set_ylabel("$U_4$")

ax1.plot(listkT, avnrg,'ro-')
ax2.plot(listkT, Cv,'bo-')
ax3.plot(listkT, avmag,'mo-')
ax4.plot(listkT, U4,'ko-')

fig.suptitle("3D Ising model (10x10x10)")
fig.tight_layout(pad=2.0)
plt.grid()
plt.savefig('3D-Ising-plots.png')
plt.show()
