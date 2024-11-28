# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Quick plotter for MLAgg DF map
Copied from 2024-07-15_Photocurrent_Plotter_v2.py

Data: 2024-08-22_BPDT_60nm_MLAgg_DF_SERS_Map.h5
    
(samples: 2024-08-20_BPDT_60nm_MLAgg_on_FTO_a)

"""

import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
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


#%% Data storage classes


class Particle(): 
    def __init__(self):
        self.name = None

class Peak():
    def __init__(self, height, mu, width, baseline, **kwargs):
        self.height = height
        self.mu = mu
        self.width = width
        self.baseline = baseline


#%% h5 files

## Load raw data h5
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-08-22.h5")

## Calibration h5 File
# cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\calibration.h5")

## h5 File for saving processed data
# save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2024-08_09_Processed_20nm_Photocurrent_Data.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory():
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\\"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% fit equations

def cottrell(x, a, b, c):
    
    return (a/(sqrt(pi*x + c))) + b

def quadratic(x, a, b, c):
    
    return a*x**2 + b*x + c


#%% Pull in darkfield spectra

parent_group = my_h5['ref_meas_0']['BPDT_MLAgg_map_0']

df_spectra = []

for group in parent_group:
    
    group = parent_group[group]
    
    for key in list(group.keys()):
        
        if 'spec' in key:
    
            spectrum = group[key]
            spectrum = df.DF_Spectrum(spectrum)
            df_spectra.append(spectrum)
      
df_avg = []
    
for spectrum in df_spectra:
    
    spectrum.y = (spectrum.y - spectrum.background)/(spectrum.reference - spectrum.background)
    spectrum.truncate(400, 900)
    df_avg.append(spectrum.y)

df_avg = df.DF_Spectrum(spectrum.x, np.mean(df_avg, axis = 0), smooth_first = False)


#%% Plot all nice

fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax.set_ylabel('Darkfield Scattering', fontsize = 'x-large')
fig.suptitle('BPDT 60 nm MLAgg Darkfield', fontsize = 'x-large')

for spectrum in df_spectra:
    ax.plot(spectrum.x, spectrum.y - spectrum.y.min(), color = 'grey', alpha = 0.1, linewidth = 5)
    
ax.plot(df_avg.x, df_avg.y - df_avg.y.min(), color = 'black', alpha = 1)

plt.tight_layout()

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'BPDT 60 nm MLAgg Darkfield' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)
    
