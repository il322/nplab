# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Co-TAPP-SMe MLAgg DF Map

Goals:
    - Plot hyperspectral map of MLAgg with thumb image & df data
    - Plot df spectra & avg for 1ML and 2ML regions
    - Plot dust map df spectra to see spot size of DF collection for 100x and 20x

Data: 2024-06-05_Co-TAPP-SME_MLAgg_DF_Map.h5


(samples:
     2024-06-04_Co-TAPP-SMe_60nm_MLAgg_on_ITO_a)

"""

from copy import deepcopy
import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
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
my_h5 = h5py.File(r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-06-05_Co-TAPP-SME_MLAgg_DF_Map.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-06-05 MLAgg DF Map Analysis\\" + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% For now pick a good group and just plot df spectra


# 20x 1ML

particle =  my_h5['PT_lab']['MLAgg_map_5']
keys = list(particle.keys())
keys = natsort.natsorted(keys)

fig, ax = plt.subplots(1,1,figsize=[16,10])
fig.suptitle('Co-TAPP-SMe 1MLAgg Darkfield 20x', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', size = 'large')
ax.set_ylabel('Scattering Intensity (au)', size = 'large')

avg_spectrum = SERS.SERS_Spectrum(x = np.zeros(1044), y = np.zeros(1044))
counter = 0
for key in keys:
    
    if 'spec' in key:
        
        spectrum = particle[key]

        spectrum = df.DF_Spectrum(spectrum)

        if int(spectrum.column) <= 3: # just take subset with only 1ML region      
            counter += 1
            spectrum.y = (spectrum.y-spectrum.background)/(spectrum.reference - spectrum.background)
            avg_spectrum.x = deepcopy(spectrum.x)
            avg_spectrum.y += spectrum.y
            ax.plot(spectrum.x, spectrum.y, color = 'lightskyblue', alpha = 0.6, zorder = 1)

avg_spectrum.y = avg_spectrum.y/counter
ax.plot(avg_spectrum.x, avg_spectrum.y, color = 'navy', zorder = 2, label = '1ML')
ax.set_xlim(400, 920)
ax.set_ylim(0,0.08)
ax.legend(fontsize = 'x-large')

plt.tight_layout(pad = 1)

## Save
save = False
if save == True:
    save_dir = r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-06-05 MLAgg DF Map Analysis\\"
    fig.savefig(save_dir + '_DF 1ML 20x' + '.svg', format = 'svg')
    plt.close(fig)


# 20x 2ML


particle =  my_h5['PT_lab']['MLAgg_map_5']
keys = list(particle.keys())
keys = natsort.natsorted(keys)

fig, ax = plt.subplots(1,1,figsize=[16,10])
fig.suptitle('Co-TAPP-SMe 2MLAgg Darkfield 20x', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', size = 'large')
ax.set_ylabel('Scattering Intensity (au)', size = 'large')

avg_spectrum = SERS.SERS_Spectrum(x = np.zeros(1044), y = np.zeros(1044))
counter = 0
for key in keys:
    
    if 'spec' in key:
        
        spectrum = particle[key]

        spectrum = df.DF_Spectrum(spectrum)

        if int(spectrum.column) >= 20 and int(spectrum.row) <= 11: # just take subset with only 1ML region      
            counter += 1
            spectrum.y = (spectrum.y-spectrum.background)/(spectrum.reference - spectrum.background)
            avg_spectrum.x = deepcopy(spectrum.x)
            avg_spectrum.y += spectrum.y
            ax.plot(spectrum.x, spectrum.y, color = 'plum', alpha = 0.6, zorder = 1)

avg_spectrum.y = avg_spectrum.y/counter
ax.plot(avg_spectrum.x, avg_spectrum.y, color = 'purple', zorder = 2, label = '2ML')
ax.set_xlim(400, 920)
ax.set_ylim(0,0.2)
ax.legend(fontsize = 'x-large')

plt.tight_layout(pad = 1)

## Save
save = False
if save == True:
    save_dir = r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-06-05 MLAgg DF Map Analysis\\"
    fig.savefig(save_dir + '_DF 2ML 20x' + '.svg', format = 'svg')
    plt.close(fig)
    
#%%   
# 100x 1ML

particle =  my_h5['PT_lab']['MLAgg_map_5']
keys = list(particle.keys())
keys = natsort.natsorted(keys)

fig, ax = plt.subplots(1,1,figsize=[16,10])
fig.suptitle('Co-TAPP-SMe 1MLAgg Darkfield 100x', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', size = 'large')
ax.set_ylabel('Scattering Intensity (au)', size = 'large')

avg_spectrum = SERS.SERS_Spectrum(x = np.zeros(1044), y = np.zeros(1044))
counter = 0
for key in keys:
    
    if 'spec' in key:
        
        spectrum = particle[key]

        spectrum = df.DF_Spectrum(spectrum)

        # if int(spectrum.column) <= 3: # just take subset with only 1ML region      
        counter += 1
        spectrum.y = (spectrum.y-spectrum.background)/(spectrum.reference - spectrum.background)
        avg_spectrum.x = deepcopy(spectrum.x)
        avg_spectrum.y += spectrum.y
        ax.plot(spectrum.x, spectrum.y, color = 'lightskyblue', alpha = 0.6, zorder = 1)

avg_spectrum.y = avg_spectrum.y/counter
ax.plot(avg_spectrum.x, avg_spectrum.y, color = 'navy', zorder = 2, label = '1ML')
ax.set_xlim(400, 920)
ax.set_ylim(0,0.2)
ax.legend(fontsize = 'x-large', loc = 'upper right')

plt.tight_layout(pad = 1)

## Save
save = False
if save == True:
    save_dir = r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-06-05 MLAgg DF Map Analysis\\"
    fig.savefig(save_dir + '_DF 1ML 100x' + '.svg', format = 'svg')
    plt.close(fig)

#%% Dust map

# 20x, 0.1um step size

particle =  my_h5['PT_lab']['dust_map_0']
keys = list(particle.keys())
keys = natsort.natsorted(keys)

fig, ax = plt.subplots(1,1,figsize=[16,10])
fig.suptitle('Dust Map Darkfield 20x', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', size = 'large')
ax.set_ylabel('Scattering Intensity (au)', size = 'large')

avg_spectrum = SERS.SERS_Spectrum(x = np.zeros(1044), y = np.zeros(1044))
counter = 0
for key in keys:
    
    if 'spec' in key:
        
        spectrum = particle[key]

        spectrum = df.DF_Spectrum(spectrum)

        if int(spectrum.column) > 10: # just take subset with only 1ML region      
            counter += 1
            spectrum.y = (spectrum.y-spectrum.background)/(spectrum.reference - spectrum.background)
            # avg_spectrum.x = deepcopy(spectrum.x)
            # avg_spectrum.y += spectrum.y
            ax.plot(spectrum.x, spectrum.y + spectrum.column/1000, color = 'lightskyblue', alpha = 0.6, zorder = 1)

avg_spectrum.y = avg_spectrum.y/counter
ax.plot(avg_spectrum.x, avg_spectrum.y, color = 'navy', zorder = 2, label = '1ML')
ax.set_xlim(400, 920)
ax.set_ylim(0,0.08)
ax.legend(fontsize = 'x-large', loc = 'upper right')

plt.tight_layout(pad = 1)

## Save
save = False
if save == True:
    save_dir = r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-06-05 MLAgg DF Map Analysis\\"
    fig.savefig(save_dir + '_DF 1ML 100x' + '.svg', format = 'svg')
    plt.close(fig)