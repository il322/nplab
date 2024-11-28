# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:39:15 2024

@author: il322


Plotter for DF of Co-TAPP-SMe MLAggs with varying 'patchiness'

- DF Map

Data: 2024-08_28_BPDT_MLAggs_FTO

(samples: 2024-08_28_BPDT_MLAggs_FTO)

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
from pybaselines.polynomial import modpoly
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

save = False
save_to_h5 = False

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

## Calibration h5 File
cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\calibration.h5")

## h5 File for saving processed data
if save_to_h5 == True:
    save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2024-09-19_Processed_Co-TAPP-SMe_Patchy_MLAgg_DF_Data.h5")

## Data h5
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-09-19_Co-TAPP-SMe_Patchy_S57nm_MLAgg_DF_Map.h5")



#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-09_19_Co-TAPP-SMe_Patchy_57nm_MLAgg_ DF Analysis\\" + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% fit equations

def exp_decay(x, m, t, b):
    return m * np.exp(-t * x)

def cottrell(x, a, b, c):
    
    return (a/(sqrt(pi*x + c))) + b

def quadratic(x, a, b, c):
    
    return a*x**2 + b*x + c

def reciprocal(x, a, b, c):
    
    return (a/(x+b)) + c

def linear(x, a, b):
    
    return a*x + b
    
#%% Darkfield

#%% DF data groups

df_h5 = my_h5

df_dict = {1 : 'Co-TAPP-SMe_60nm_MLAgg_1_1x_0', 
           0.9 : 'Co-TAPP-SMe_60nm_MLAgg_2_0,9x_0', 
           0.8 : 'Co-TAPP-SMe_60nm_MLAgg_3_0,8x_0', 
           0.7 : 'Co-TAPP-SMe_60nm_MLAgg_4_0,7x_0'}

concs = [1, 0.9, 0.8, 0.7]

sizes = [57]


#%% Get all DF spectra


df_spectra_dict = {1 : [], 
           0.9 : [], 
           0.8 : [], 
           0.7 : []}


df_avg_dict = {1 : [], 
           0.9 : [], 
           0.8 : [], 
           0.7 : []}

df_sem_dict = {1 : [], 
           0.9 : [], 
           0.8 : [], 
           0.7 : []}


for conc in concs:
    
    df_spectra = df_spectra_dict[conc]
    df_avg = df_avg_dict[conc]
    scan = df_h5['PT_lab'][df_dict[conc]]
    
    for key in list(scan.keys()):
        
        group = scan[key]
        
        for key in list(group.keys()):

            if 'spec' in key:
                spectrum = group[key]
                spectrum = df.DF_Spectrum(spectrum)
                spectrum.y = (spectrum.y - spectrum.background)/(spectrum.reference - spectrum.background)
                spectrum.truncate(400, 900)
                if spectrum.y.min() < 0:
                    continue
                # if spectrum.y[0] > 0.05:
                #     print(key)
                df_spectra.append(spectrum)
                df_avg.append(spectrum.y)

    df_sem = df.DF_Spectrum(spectrum.x, np.std(df_avg, axis = 0))
    df_avg = df.DF_Spectrum(spectrum.x, np.mean(df_avg, axis = 0), smooth_first = False)
    df_avg_dict[conc] = df_avg
    df_sem_dict[conc] = df_sem


#%% Plot all df spectra


my_cmap = plt.get_cmap('viridis_r')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=0.65, vmax=1)

fig, ax = plt.subplots(1, 1, figsize=[16,10])
ax.set_xlabel('Wavelength (nm)', fontsize = 'x-large')
ax.set_ylabel('Darkfield Scattering', fontsize = 'x-large')
fig.suptitle('Patchy Co-TAPP-SMe 57nm MLAgg on FTO Darkfield', fontsize = 'x-large')

for i, size in enumerate(concs):
    
    df_spectra = df_spectra_dict[size]

    spectrum = df_avg_dict[size]    
    sem = df_sem_dict[size]
    
    color = cmap(norm(size))  
    
    ax.plot(spectrum.x, spectrum.y, color = color, alpha = 1, label = str(size) + 'x Concentration', zorder = 1)
    ax.fill_between(sem.x, spectrum.y - sem.y, spectrum.y + sem.y, color = color, alpha = 0.05, zorder = 0)  

ax.legend(loc = 'upper left')

## Save
if save == True:
    save_dir = get_directory('DF Compiled')
    fig.savefig(save_dir + 'Patchy Co-TAPP-SMe 57nm MLAgg DF' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)

if save_to_h5 == True:
    group = save_h5

    canvas = FigureCanvas(fig)
    canvas.draw()
    fig_array = np.array(canvas.renderer.buffer_rgba())
    group.create_dataset(name = 'DF_Compiled_jpg_%d', data = fig_array)