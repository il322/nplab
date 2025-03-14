# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:05:48 2023

@author: il322
"""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl


''' 
    Quick plotter for laser safety stuff in Lab 1 NKT rig
    Refer to OneNote onenote:///O:\il322\il322%20Notebook\2024\July.one#10/07/2024&section-id={3A78314B-E628-4CD4-9A49-8BDAF06E0FE3}&page-id={91777A79-B53E-4964-B1EB-74C8437976B3}&end
    for MPE values & formulas
'''

plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)


#%% Plot irradiance v laser output power with MPE hlines

# Calculate actual laser irradiance in W/m2
power = np.linspace(0, 400, 1001)
I = (power/1000)/(3.85*(10**-5))
MPE = 0.004

# Plotting
fig = plt.figure(figsize=(8,6))
plt.plot(power, I, label = 'Laser Irradiance')
plt.title('Laser Irradiance v. Output Power')
plt.xlabel('Laser Output Power (mW)')
plt.ylabel('Laser Irradiance (Wm$^{-2}$)')
## Plot MPE h_lines for MPE_633 (0.25s) and MPE_785 & MPE_852 (10s)
plt.hlines(MPE, 0, power.max(), color = 'red', linestyles='dashed', label = 'MPE NKT, 0.25s')
plt.xlim(0, 0.1)
plt.ylim(0, 0.1)
plt.legend(loc=2)
plt.show()

#%% Plot scattering NOHD v laser output power
power = np.linspace(0, 100, 1001)
# Calculate NOHD (cm) using 100s exposure and 2 rad divergence
NOHD = (((4 * power / 1000)/(np.pi * 0.0001/0.25))**(1/2) - (0.7/1000)) / 2 * 100

# Plotting
fig = plt.figure(figsize=(8,6))
plt.plot(power, NOHD)
plt.title('Scattered Light NOHD v. Output Power')
plt.xlabel('Laser Output Power (mW)')
plt.ylabel('Nominal Ocular Hazard Distance (cm)')
# plt.legend(loc=1)
plt.show()