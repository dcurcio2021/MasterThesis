# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:04:31 2024

@author: curcio_d
"""

import math

def kon(alpha, D, e, w):
    ex = e / D
    f = math.pi / 180.0

    alpha = alpha % 360.0
    if alpha > 180.0:
        alpha = 360.0 - alpha

    a = 0
    b = ex * 0.035
    a2 = a * a
    b2 = b * b
    si = math.sin(alpha * f)
    co = math.cos(alpha * f)
    si2 = si * si
    co2 = co * co

    x = a * co + b * si + math.sqrt(2.0 * a * b * co * si + co2 + si2 - a2 * si2 - b2 * co2)
    x = (x - 1.0) * D
    x = max(x, 0.0)
    
    return x

#%% Lager Mitte

def main():
    D = 0.18  # Lagerdurchmesser m
    e = 0.001 # Exzentrizität 
    w = 2.0   # Winkel
    name = 'U:\\First Simulation\\buchse_innen_mitte.vrt'
    output_file = 'U:\\First Simulation\\contourLM35mu.dat'

    with open(name, 'r') as infile, open(output_file, 'w') as outfile:
        # Skip the first two lines
        infile.readline()
        infile.readline()

        for line in infile:
            iv, x, y, z = map(float, line.split())
            alpha = math.atan2(x, y) * 180.0 / math.pi
            ko = kon(alpha, D, e, w)
            ko += 90.0 / 1000000.0
            print(f"{x} {y} {alpha} {ko}")
            outfile.write(f"{int(iv)} {ko}\n")

if __name__ == "__main__":
    main()
#%% Lager Aussen

def main():
    D = 0.18  # Lagerdurchmesser m
    e = 0.001 # Exzentrizität 
    w = 2.0   # Winkel
    name = 'U:\\First Simulation\\buchse_innen.vrt'
    output_file = 'U:\\First Simulation\\contourLA35mu.dat'

    with open(name, 'r') as infile, open(output_file, 'w') as outfile:
        # Skip the first two lines
        infile.readline()
        infile.readline()

        for line in infile:
            iv, x, y, z = map(float, line.split())
            alpha = math.atan2(x, y) * 180.0 / math.pi
            ko = kon(alpha, D, e, w)
            ko += 90.0 / 1000000.0
            print(f"{x} {y} {alpha} {ko}")
            outfile.write(f"{int(iv)} {ko}\n")

if __name__ == "__main__":
    main()
    