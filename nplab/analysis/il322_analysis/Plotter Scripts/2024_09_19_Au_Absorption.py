# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 07:52:06 2024

@author: il322

Messing around with absorption in thin film Au

"""

from copy import deepcopy
import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvas
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import statistics
import scipy
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter
from scipy.stats import norm
from tqdm import tqdm
from pylab import *
import nplab
import h5py
import natsort
import os

from nplab import datafile
from nplab.analysis.general_spec_tools import spectrum_tools as spt
from nplab.analysis.general_spec_tools import npom_sers_tools as nst
from nplab.analysis.general_spec_tools import agg_sers_tools as ast
from nplab.analysis.SERS_Fitting import Auto_Fit_Raman as afr
from nplab.analysis.il322_analysis import il322_calibrate_spectrum as cal
from nplab.analysis.il322_analysis import il322_SERS_tools as SERS
from nplab.analysis.il322_analysis import il322_DF_tools as df

from lmfit.models import GaussianModel


#%%

au = np.genfromtxt(r"C:\Users\il322\Desktop\Offline Data\Yakubovsky-117nm.txt", skip_header = 1)
wln = au[:,0] * 1E-6 ## wln in m
n = au[:,1]
k = au[:,2]
nk = n+(1j*k)

## Abs from alpha
alpha = (4 * np.pi * k)/(wln) # alpha in 1/m
A = (1 - np.exp(-1 * alpha * 60E-9)) * 100 # Abs = 100% * [1 - e^(-alpha * d)]; d = 60 nm

## Reflectance
R = np.abs((1 - nk)/(1 + nk))**2 # R at top interface
R2 = np.abs((1 - nk)/(1 + nk))**2 # R at bottom interface
thick = 60E-9 # thickness of Au in m

T_1 = (1 - R) # I after 1st air-Au interface
T_2 = (1 - R) * e**(-alpha * thick) # I after Au thin film of thickness = thick 
T_3 = (1 - R2) * T_2 # I after 2nd Au-air interface
A = T_1 - T_3 # Absorption in thin film approximation

''' Note that A doesn't take into account absorption from cascading back reflections at Au interfaces'''

fig, ax = plt.subplots(1, 1, figsize=[10,8])

ax.plot(wln*1E9, T_1, label = 'T after Au-air (1-R)')
ax.plot(wln*1E9, T_2, label = 'T after Au thin film (1-R) * A_film')
ax.plot(wln*1E9, T_3, label = 'T after Au film back into air (1-R2) * (1-R) * A_film')
ax.plot(wln*1E9, A, label = 'Final Abs')
ax.set_ylabel('I/I$_0$')
ax.set_xlabel('Wavelength (nm)')
ax.set_yscale('linear')
ax.set_xlim(400, 1000)
ax.legend(fontsize = 10)
ax.set_title('Au thin film absorption (thickness = ' + str(np.round(thick* 1E9)) + ' nm)', fontsize = 15)

plt.show()

#%%

plt.plot(wln*1E9, 1 - R)
plt.xlim(350, 800)
plt.ylabel('1 - Reflactance')

fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax.plot(wln*1E9, alpha, color = 'red')
ax2 = twinx(ax)
ax2.plot(wln*1E9, A, color = 'black')



ax.set_ylabel('Absorption Coefficient', color = 'red')
ax2.set_ylabel('Absorption (%)')
ax.set_xlabel('Wavelength (nm)')
# plt.plot(wln, (e**(-60E-9 * alpha)))
plt.xlim(400, 1000)