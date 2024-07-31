# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Quick plotter for some photocurrent measurements
Not much analysis and not really process streamlined

Copied from 2024-06-11_Photocurrent_Plotter.py

Data: 2024-06-28_ITO_Co-TAPP-SMe_MLAgg_Photocurrent.h5


(samples: 2024-06-04_Co-TAPP-SMe_60nm_MLAgg_on_ITO_f
          2024-07-01_Bare_ITO)

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
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-06-28_ITO_Co-TAPP-SMe_MLAgg_Photocurrent.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-06-28 Photocurrent Analysis\\" + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%% Plot Dark/Light LSV

# particle = my_h5['Potentiostat']
# # particle = Particle()
# particle_name = 'Co-TAPP-SMe 60nm MLAgg on ITO'


# # Dark v Light LSV plots

# ## Get all spectra to plot with times
# keys = list(particle.keys())
# keys = natsort.natsorted(keys)
# spectra = []
# timestamps = []
# for key in keys:    
#     if 'lsv' in key and key != 'dark_lsv_1' and key != 'dark_lsv_9' and key != 'dark_lsv_0':
#         spectrum = particle[key]
#         timestamp = spectrum.attrs['creation_timestamp']       
#         spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
#         spectrum.timestamp = timestamp[timestamp.find('T')+1:]
#         spectrum.name = key
#         spectra.append(spectrum)
        
# ## Sort spectra by time
# spectra = natsort.natsorted(spectra, key = lambda x : x.timestamp)

# ## Plot dark & LSV in time order
# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_xlabel('Potential v. Ag/AgCl (V)')
# ax.set_ylabel('Current ($\mu$A)')
# ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Dark LSV')
# offset = 0
# for i, spectrum in enumerate(spectra):
#     if 'dark' in spectrum.name:
#         color = 'grey'
#         label = 'Dark'
#     else:
#         color = 'orange'
#         label = 'Light'

#     ax.plot(spectrum.x, spectrum.y + (i * offset), label = i, color = color, alpha =  (i+1)/(len(spectra)))
    
# ax.set_xlim(-0.41, 0.21)
# ax.set_ylim(-10, 15)
# ax.legend(title = 'LSV Order', ncol = 3)

# save = False
# if save == True:
#     save_dir = get_directory(particle.name)
#     fig.savefig(save_dir + particle_name + 'LSV' + '.svg', format = 'svg')
#     plt.close(fig)
    

#%% Plot 10s Light Toggle Photocurrent

particle = my_h5['Potentiostat']
particle_name = 'Co-TAPP-SMe 60nm MLAgg on ITO'

skip_list = ['Co-TAPP-SMe_toggle_10s_CA_0.0V_0',
             'Co-TAPP-SMe_toggle_10s_CA_0.3V_0',
              'Co-TAPP-SMe_dark_CA_0.2V_0',
              # 'Co-TAPP-SMe_dark_CA_0.2V_1',
              # 'Co-TAPP-SMe_toggle_10s_CA_0.2V_0',
             'Co-TAPP-SMe_dark_CA_0.3V_0',
             'Co-TAPP-SMe_dark_CA_0.3V_1',
             'Co-TAPP-SMe_dark_CA_-0.4V_0',
              # 'ITO_toggle_10s_CA_0.2V_0',
             'ITO_dark_CA_0.2V_0']


# Get all spectra to plot with times

keys = list(particle.keys())
keys = natsort.natsorted(keys)
spectra = []
timestamps = []
for key in keys:    
    if ('Co-TAPP' in key or 'ITO' in key) and key not in skip_list:
        if 'toggle_10s' in key or 'dark_CA' in key and 'ITO_dark' not in key:
            print(key)
            spectrum = particle[key]
            timestamp = spectrum.attrs['creation_timestamp']
            voltage = spectrum.attrs['Levels_v (V)']
            spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
            spectrum.timestamp = timestamp[timestamp.find('T')+1:]
            spectrum.name = key
            ### Truncate & baseline
            spectrum.voltage = voltage[0]
            spectrum.truncate(40, 180)
            spectrum.y -= spt.baseline_als(spectrum.y, 1e5, 1e-3)
            spectrum.y = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 3000, fs = 60000)
            if 'Co-TAPP' in key:
                spectrum.sample = 'MLAgg'
            elif 'ITO' in key:
                spectrum.sample = 'ITO'
            spectra.append(spectrum)
            
            if 'dark' in key:
                spectrum.dark = True
            else:
                spectrum.dark = False
            

## Sort spectra by voltage (for color mapping)
spectra = natsort.natsorted(spectra, key = lambda x : x.voltage)


# Plot

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A) (offset)')
ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Light Toggle 10s')
offset = 0.02
my_cmap = plt.get_cmap('viridis')


for i, spectrum in enumerate(spectra):

    if spectrum.sample == 'ITO':
        linetype = 'dashed'
    else:
        linetype = 'solid'    
        
    if spectrum.dark == True:
        linetype = 'solid'
        alpha = 0.3
    else:
        alpha = 1

    color = my_cmap((spectrum.voltage - -0.4) * 1.5)   
    ax1 = ax.plot(spectrum.x, (spectrum.y + spectrum.voltage * 10 * offset) + 0.08, label = spectrum.voltage, color = color, alpha = alpha, linestyle = linetype)

## Ax lims
# ax.set_xlim(40, 180)
# ax.set_ylim(-0.001, 0.2)
# ax.set_yscale('symlog')
# ylim = ax.get_ylim()

## On/Off bars
# bars = np.arange(0, int(spectrum.x.max()), 20)
# ax.bar(bars, 1, width = 10, align = 'edge', zorder = 1, color = 'grey', alpha = 0.5, label = 'Dark')
# ax.bar(bars + 10, 1, width = 10, align = 'edge', zorder = 1, color = 'white', alpha = 1, label = 'Light')

## Legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
leg1 = ax.legend(reversed(by_label.values()), reversed(by_label.keys()), title = 'Potential v.\nAg/AgCl (V)', ncol = 1, loc = 'upper right')
ax.add_artist(leg1)

x = ax.plot(100,0, color = 'black', alpha = 0.3, linestyle = 'solid', label = 'Dark')
y = ax.plot(100,0, color = 'black', alpha = 1, linestyle = 'dashed', label = 'ITO')
z = ax.plot(100,0, color = 'black', alpha = 1, linestyle = 'solid', label = 'MLAgg')
leg2 = ax.legend(handles = [z[0],y[0],x[0]], loc = 'upper left')
ax.add_artist(leg2)

plt.tight_layout()

## Save
save = True
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + '10s Toggle Photocurrent CA' + '.svg', format = 'svg')
    plt.close(fig)



#%% Plot 30s Light Toggle Photocurrent

particle = my_h5['Potentiostat']
particle_name = 'Co-TAPP-SMe 60nm MLAgg on ITO'

skip_list = ['Co-TAPP-SMe_toggle_10s_CA_0.0V_0',
             'Co-TAPP-SMe_toggle_10s_CA_0.3V_0',
              'Co-TAPP-SMe_dark_CA_0.2V_0',
              # 'Co-TAPP-SMe_dark_CA_0.2V_1',
              # 'Co-TAPP-SMe_toggle_10s_CA_0.2V_0',
             'Co-TAPP-SMe_dark_CA_0.3V_0',
             'Co-TAPP-SMe_dark_CA_0.3V_1',
             'Co-TAPP-SMe_dark_CA_-0.4V_0',
              # 'ITO_toggle_30s_CA_-0.4V_0',
             'ITO_dark_CA_0.2V_0',
             'ITO_toggle_30s_CA_0.0V_0',
             'ITO_toggle_30s_CA_0.0V_1']


# Get all spectra to plot with times

keys = list(particle.keys())
keys = natsort.natsorted(keys)
spectra = []
timestamps = []
for key in keys:    
    if ('Co-TAPP' in key or 'ITO' in key) and key not in skip_list:
        if 'toggle_30s' in key or 'dark_CA' in key and 'ITO_dark' not in key:
            spectrum = particle[key]
            timestamp = spectrum.attrs['creation_timestamp']
            voltage = spectrum.attrs['Levels_v (V)']
            spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
            spectrum.timestamp = timestamp[timestamp.find('T')+1:]
            spectrum.name = key
            ### Truncate & baseline
            spectrum.voltage = voltage[0]
            spectrum.truncate(40, 180)
            spectrum.y -= spt.baseline_als(spectrum.y, 1e5, 1e-4)
            spectrum.y = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 3000, fs = 60000)
            if 'Co-TAPP' in key:
                spectrum.sample = 'MLAgg'
            elif 'ITO' in key:
                spectrum.sample = 'ITO'
            spectra.append(spectrum)
            
            if 'dark' in key:
                spectrum.dark = True
            else:
                spectrum.dark = False
            

## Sort spectra by voltage (for color mapping)
spectra = natsort.natsorted(spectra, key = lambda x : x.voltage)


# Plot

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A) (offset)')
ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Light Toggle 30s')
offset = 0.02
my_cmap = plt.get_cmap('viridis')


for i, spectrum in enumerate(spectra):

    if spectrum.sample == 'ITO':
        linetype = 'dashed'
    else:
        linetype = 'solid'    
        
    if spectrum.dark == True:
        linetype = 'solid'
        alpha = 0.3
    else:
        alpha = 1

    color = my_cmap((spectrum.voltage - -0.4) * 1.5)   
    ax1 = ax.plot(spectrum.x, (spectrum.y + spectrum.voltage * 10 * offset) + 0.08, label = spectrum.voltage, color = color, alpha = alpha, linestyle = linetype)

## Ax lims
# ax.set_xlim(40, 180)
# ax.set_ylim(-0.05, 0.0)
# ax.set_yscale('symlog')
# ylim = ax.get_ylim()

## On/Off bars
# bars = np.arange(0, int(spectrum.x.max()), 20)
# ax.bar(bars, 1, width = 10, align = 'edge', zorder = 1, color = 'grey', alpha = 0.5, label = 'Dark')
# ax.bar(bars + 10, 1, width = 10, align = 'edge', zorder = 1, color = 'white', alpha = 1, label = 'Light')

## Legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
leg1 = ax.legend(reversed(by_label.values()), reversed(by_label.keys()), title = 'Potential v.\nAg/AgCl (V)', ncol = 1, loc = 'upper right')
ax.add_artist(leg1)
x = ax.plot(100,0, color = 'black', alpha = 0.3, linestyle = 'solid', label = 'Dark')
y = ax.plot(100,0, color = 'black', alpha = 1, linestyle = 'dashed', label = 'ITO')
z = ax.plot(100,0, color = 'black', alpha = 1, linestyle = 'solid', label = 'MLAgg')
leg2 = ax.legend(handles = [z[0],y[0],x[0]], loc = 'upper left')
ax.add_artist(leg2)

plt.tight_layout()

## Save
save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + '30s Toggle Photocurrent CA' + '.svg', format = 'svg')
    plt.close(fig)
 

#%%
    
# Plot photocurrent difference from 30s toggle CA

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Potential v. Ag/AgCl(V)')
ax.set_ylabel('Photocurrent (nA)')
ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Light Toggle 30s')
my_cmap = plt.get_cmap('viridis')
for spectrum in spectra:
    
    color = my_cmap((spectrum.voltage - -0.4) * 1.5) 
    
    if spectrum.sample == 'ITO':
        marker = 'd'
        label = 'ITO'
    elif spectrum.dark == True:
        marker = 'x'
        label = 'Dark'
    else:
        marker = 'o'
        label = 'MLAgg'
        
    # if spectrum.dark == True:
    #     linetype = 'solid'
    #     alpha = 0.3
    # else:
    #     alpha = 1
    
    ## Plot current at 180s - current at 150s
    current_180 = np.mean(spectrum.y[1390:1399])
    current_150 = np.mean(spectrum.y[1090:1100])
    ax.scatter(spectrum.voltage, (current_180 - current_150)*10**3, label = label, color = color, s = 100, marker = marker)
    
    ## Plot current at 120s - current at 90s
    current_120 = np.mean(spectrum.y[790:800])
    current_90 = np.mean(spectrum.y[490:500])
    ax.scatter(spectrum.voltage, (current_120 - current_90)*10**3, label = label, color = color, s = 100, marker = marker)

# if sample.voltage == -0.1:
#     print(current_180 - current_150)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(reversed(by_label.values()), reversed(by_label.keys()), ncol = 3)   
leg = ax.get_legend()
for handle in ax.get_legend().legendHandles:
    handle.set_color('black') 

plt.tight_layout()

## Save
save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + '30s Photocurrent' + '.svg', format = 'svg')
    plt.close(fig)
    

    