# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Quick plotter for some photocurrent measurements
Not much analysis and not really process streamlined

Data: 2024-05-21.h5


(samples:
)

"""

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
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-05-21.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r'C:\Users\il322\Desktop\Offline Data\2024-05-10_MLAgg Photocurrent Analysis\\' + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%% CB5 Plotting

particle = my_h5['CB5_MLAgg_0']
# particle = Particle()
particle_name = 'CB5_MLAgg_0'


# Dark v Light LSV plots

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Potential v. Ag/AgCl (V)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('CB[5] MLAgg Photocurrent - LSV')

spectrum = particle['dark_LSV_3']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, label = 'Dark', color = 'grey')

spectrum = particle['light_LSV_0']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, label = 'Light', color = 'orange')

spectrum = particle['dark_LSV_4']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'grey')

spectrum = particle['light_LSV_1']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'orange')

ax.legend()

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'LSV' + '.svg', format = 'svg')
    plt.close(fig)
    
    
# CA Light toggle plots

## 10s toggle
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('CB[5] MLAgg Photocurrent - CA @ -0.4V')

spectrum = particle['light_toggle_10s_CA_-0.4V_0']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'green', zorder = 2)
ax.set_ylim(-0.5, -0.3)
ylim = ax.get_ylim()
bars = np.arange(0, int(spectrum.x.max()), 20)
ax.bar(bars, -10, width = 10, align = 'edge', zorder = 1, color = 'black', alpha = 0.1, label = 'Dark')
ax.bar(bars + 10, -10, width = 10, align = 'edge', zorder = 1, color = 'yellow', alpha = 0.1, label = 'Light')
# spectrum = particle['dark_LSV_4']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'black')

# spectrum = particle['light_LSV_1']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'red')

ax.legend()

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'CA_-0.4V_10s' + '.svg', format = 'svg')
    plt.close(fig)

## 30s toggle
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('CB[5] MLAgg Photocurrent - CA @ -0.4V')

spectrum = particle['light_toggle_30s_CA_-0.4V_0']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'green', zorder = 2)
ax.set_ylim(-0.5, -0.3)
ylim = ax.get_ylim()
bars = np.arange(0, int(spectrum.x.max()), 60)
ax.bar(bars, -10, width = 30, align = 'edge', zorder = 1, color = 'black', alpha = 0.1, label = 'Dark')
ax.bar(bars + 30, -10, width = 30, align = 'edge', zorder = 1, color = 'yellow', alpha = 0.1, label = 'Light')
# spectrum = particle['dark_LSV_4']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'black')

# spectrum = particle['light_LSV_1']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'red')

ax.legend()

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'CA_-0.4V_30s' + '.svg', format = 'svg')
    plt.close(fig)
    
    
    
## 0V 10s toggle
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('CB[5] MLAgg Photocurrent - CA @ 0V')

spectrum = particle['light_toggle_10s_CA_0V_0']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'green', zorder = 2)
ax.set_ylim(-0.0075, -0.0025)
ylim = ax.get_ylim()
bars = np.arange(0, int(spectrum.x.max()), 20)
ax.bar(bars, -10, width = 10, align = 'edge', zorder = 1, color = 'black', alpha = 0.1, label = 'Dark')
ax.bar(bars + 10, -10, width = 10, align = 'edge', zorder = 1, color = 'yellow', alpha = 0.1, label = 'Light')
# spectrum = particle['dark_LSV_4']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'black')

# spectrum = particle['light_LSV_1']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'red')

ax.legend()

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'CA_0V_10s' + '.svg', format = 'svg')
    plt.close(fig)


#%% BPT Plotting

particle = my_h5['BPT_MLAgg_0']
# particle = Particle()
particle_name = 'BPT_MLAgg_0'


# Dark v Light LSV plots

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Potential v. Ag/AgCl (V)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('BPT MLAgg Photocurrent - LSV')

spectrum = particle['dark_LSV_10']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, label = 'Dark', color = 'grey')

spectrum = particle['light_LSV_3']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, label = 'Light', color = 'orange')

spectrum = particle['dark_LSV_11']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'grey')

spectrum = particle['light_LSV_4']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'orange')

ax.legend()

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'LSV' + '.svg', format = 'svg')
    plt.close(fig)
    

# CA Light toggle plots

## 10s toggle
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('BPT MLAgg Photocurrent - CA @ -0.4V')

spectrum = particle['light_toggle_10s_CA_-0.4V_1']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'green', zorder = 2)
ax.set_ylim(-2.6, -2)
ax.set_xlim(25, None)
ylim = ax.get_ylim()
bars = np.arange(0, int(spectrum.x.max()), 20)
ax.bar(bars, -10, width = 10, align = 'edge', zorder = 1, color = 'black', alpha = 0.1, label = 'Dark')
ax.bar(bars + 10, -10, width = 10, align = 'edge', zorder = 1, color = 'yellow', alpha = 0.1, label = 'Light')
# spectrum = particle['dark_LSV_4']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'black')

# spectrum = particle['light_LSV_1']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'red')

ax.legend()

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'CA_-0.4V_10s' + '.svg', format = 'svg')
    plt.close(fig)

## 30s toggle
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('BPT MLAgg Photocurrent - CA @ -0.4V')

spectrum = particle['light_toggle_30s_CA_-0.4V_0']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'green', zorder = 2)
ax.set_ylim(-2.6, -2)
ax.set_xlim(25, None)
ylim = ax.get_ylim()
bars = np.arange(0, int(spectrum.x.max()), 60)
ax.bar(bars, -10, width = 30, align = 'edge', zorder = 1, color = 'black', alpha = 0.1, label = 'Dark')
ax.bar(bars + 30, -10, width = 30, align = 'edge', zorder = 1, color = 'yellow', alpha = 0.1, label = 'Light')
# spectrum = particle['dark_LSV_4']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'black')

# spectrum = particle['light_LSV_1']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'red')

ax.legend()

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'CA_-0.4V_30s' + '.svg', format = 'svg')
    plt.close(fig)
    
## 0V 10s toggle
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('BPT MLAgg Photocurrent - CA @ 0V')

spectrum = particle['light_toggle_10s_CA_0V_2']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
ax.plot(spectrum.x, spectrum.y, color = 'green', zorder = 2)
ax.set_ylim(0.25, 0.4)
ax.set_xlim(10, None)
ylim = ax.get_ylim()
bars = np.arange(0, int(spectrum.x.max()), 20)
ax.bar(bars, 10, width = 10, align = 'edge', zorder = 1, color = 'black', alpha = 0.1, label = 'Dark')
ax.bar(bars + 10, 10, width = 10, align = 'edge', zorder = 1, color = 'yellow', alpha = 0.1, label = 'Light')
# spectrum = particle['dark_LSV_4']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'black')

# spectrum = particle['light_LSV_1']
# spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
# ax.plot(spectrum.x, spectrum.y, color = 'red')

ax.legend()

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'CA_0V_10s' + '.svg', format = 'svg')
    plt.close(fig)
    