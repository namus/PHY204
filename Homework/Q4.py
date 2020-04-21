#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:39:01 2019

@author: Suman
"""
import numpy as np

# Start reading data from file ...
fo = open("unknown.xyz", "r")

numatom = int(fo.readline()) # Read number of atoms from first line
fo.readline() # Skip second line as dummy

atomname = [] # Atom names will be saved here
coord = np.zeros((numatom,3)) # Coordinates will be saved here

for i in range(numatom):
    line = fo.readline().split()
    atomname.append(line[0])
    coord[i][0] = float(line[1])
    coord[i][1] = float(line[2])
    coord[i][2] = float(line[3])
fo.close()

# Identifying bonds starts ...
bondlist = [] # It will contain number of bonds formed by each atom i
totalbond = 0
for i in range(numatom):
    countbond = 0 # This is number of bonds for each atom i
    for j in range(numatom):
        if i != j:
            dist = np.linalg.norm(coord[i] - coord[j])
            if dist < 1.6:
                countbond += 1
                totalbond += 1 # Each bond is double counted!
    bondlist.append(countbond)

# Print TOTAL number of bonds in the molecule
print(f"Total number of bonds is: {totalbond//2}") # Each bond is double counted!

# Dictionary definiting the atom identity in terms of number of bonds
bondmap = {1: 'H', 2: 'O', 3: 'N', 4: 'C'} 

# "set" has unique members from a list (non-repeating)
# Here we print the count of each atom type in molecule
# bondlist.count(i) gives the count of "i" in the list "bondlist"
for i in set(bondlist):
    print(f"Number of {bondmap[i]} atoms = {bondlist.count(i)}")

# Write the coordinates with correct atom names to solution.xyz file ...
fo = open("solution.xyz", "w")
fo.write(str(numatom)+"\n")
fo.write("Solution!\n")
for i in range(numatom):
    line = f"{bondmap[bondlist[i]]} \t  {coord[i][0]} \t  {coord[i][1]} \t {coord[i][2]}\n"
    fo.write(line)
fo.close()
print("Coordinates with correct atom names have been written to solution.xyz!")

