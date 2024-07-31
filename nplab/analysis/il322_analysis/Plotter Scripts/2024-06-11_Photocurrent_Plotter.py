# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Quick plotter for some photocurrent measurements
Not much analysis and not really process streamlined

Copied from 2024-05-21_Photocurrent_Plotter.py

Data: 2024-06-11.h5


(samples: 2024-06-04_Co-TAPP-SMe_60nm_MLAgg_on_ITO_e)

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
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-06-11_Photocurrent_Co-TAPP-SMe_MLAgg.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-06-11 Photocurrent Analysis\\" + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%% Plot Dark/Light LSV

particle = my_h5['Potentiostat']
# particle = Particle()
particle_name = 'Co-TAPP-SMe 60nm MLAgg on ITO'


# Dark v Light LSV plots

## Get all spectra to plot with times
keys = list(particle.keys())
keys = natsort.natsorted(keys)
spectra = []
timestamps = []
for key in keys:    
    if 'lsv' in key and key != 'dark_lsv_1' and key != 'dark_lsv_9' and key != 'dark_lsv_0':
        spectrum = particle[key]
        timestamp = spectrum.attrs['creation_timestamp']       
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = key
        spectra.append(spectrum)
        
## Sort spectra by time
spectra = natsort.natsorted(spectra, key = lambda x : x.timestamp)

## Plot dark & LSV in time order
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Potential v. Ag/AgCl (V)')
ax.set_ylabel('Current ($\mu$A)')
ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Dark LSV')
offset = 0
for i, spectrum in enumerate(spectra):
    if 'dark' in spectrum.name:
        color = 'grey'
        label = 'Dark'
    else:
        color = 'orange'
        label = 'Light'

    ax.plot(spectrum.x, spectrum.y + (i * offset), label = i, color = color, alpha =  (i+1)/(len(spectra)))
    
ax.set_xlim(-0.41, 0.21)
ax.set_ylim(-10, 15)
ax.legend(title = 'LSV Order', ncol = 3)

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + 'LSV' + '.svg', format = 'svg')
    plt.close(fig)
    

#%% Plot 10s Light Toggle Photocurrent

particle = my_h5['Potentiostat']
# particle = Particle()
particle_name = 'Co-TAPP-SMe 60nm MLAgg on ITO'


# 10s Light toggle photocurrent plots at 0.0V to -0.4V

## Get all spectra to plot with times
keys = list(particle.keys())
keys = natsort.natsorted(keys)
spectra = []
timestamps = []
for key in keys:    
    if 'light_toggle_10s' in key and key != 'light_toggle_10s_CA_0.0V_20' and key != 'light_toggle_10s_CA_0.0V_0' and key != 'light_toggle_10s_CA_-0.3V_0' and key != 'light_toggle_10s_CA_-0.4V_1':
        spectrum = particle[key]
        timestamp = spectrum.attrs['creation_timestamp']
        voltage = spectrum.attrs['Levels_v (V)']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = key
        spectrum.voltage = voltage[0]
        spectrum.truncate(40, 180)
        spectra.append(spectrum)
        
## Sort spectra by time
# spectra = natsort.natsorted(spectra, key = lambda x : x.timestamp)

## Plot dark & LSV in time order
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A) (offset)')
ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Light Toggle 10s')
offset = 0
my_cmap = plt.get_cmap('viridis')
for i, spectrum in enumerate(spectra):
    # if 'dark' in spectrum.name:
    #     color = 'grey'
    #     label = 'Dark'
    # else:
    #     color = 'orange'
    #     label = 'Light'
    
    
    # norm = mpl.colors.Normalize(vmin=-0.4, vmax=0)
    color = my_cmap(spectrum.voltage * -2.5)
    

    
    if spectrum.voltage < 0.1:
        # spectrum.y = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1500, fs = 60000)
        ax.plot(spectrum.x, spectrum.y - spectrum.y.min(), label = spectrum.voltage, color = color)
        print(-spectrum.voltage * 10)
    
ax.set_xlim(40, 180)
ax.set_ylim(-0.001, 0.1)
# ax.set_yscale('symlog')
# ax.legend(title = 'Potential v. Ag/AgCl (V)', ncol = 3)
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), title = 'Potential v. Ag/AgCl (V)', ncol = 2)

ylim = ax.get_ylim()
bars = np.arange(0, int(spectrum.x.max()), 20)
ax.bar(bars, 1, width = 10, align = 'edge', zorder = 1, color = 'grey', alpha = 0.5, label = 'Dark')
ax.bar(bars + 10, 1, width = 10, align = 'edge', zorder = 1, color = 'white', alpha = 1, label = 'Light')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title = 'Potential v. Ag/AgCl (V)', ncol = 2, facecolor = 'lightgrey')

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + '10s Toggle Photocurrent CA' + '.svg', format = 'svg')
    plt.close(fig)



#%% Plot 30s Light Toggle Photocurrent

particle = my_h5['Potentiostat']
# particle = Particle()
particle_name = 'Co-TAPP-SMe 60nm MLAgg on ITO'


# 10s Light toggle photocurrent plots at 0.0V to -0.4V

## Get all spectra to plot with times
keys = list(particle.keys())
keys = natsort.natsorted(keys)
spectra = []
timestamps = []
for key in keys:    
    if 'light_toggle_30s' in key and key != 'light_toggle_10s_CA_0.0V_20' and key != 'light_toggle_10s_CA_0.0V_0' and key != 'light_toggle_10s_CA_-0.3V_0' and key != 'light_toggle_10s_CA_-0.4V_1':
        spectrum = particle[key]
        timestamp = spectrum.attrs['creation_timestamp']
        voltage = spectrum.attrs['Levels_v (V)']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = key
        spectrum.voltage = voltage[0]
        spectrum.truncate(40, 180)
        spectra.append(spectrum)
        
## Sort spectra by time
# spectra = natsort.natsorted(spectra, key = lambda x : x.timestamp)

## Plot dark & LSV in time order
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Current ($\mu$A) (offset)')
ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Light Toggle 30s')
offset = 0
my_cmap = plt.get_cmap('viridis')
for i, spectrum in enumerate(spectra):
    # if 'dark' in spectrum.name:
    #     color = 'grey'
    #     label = 'Dark'
    # else:
    #     color = 'orange'
    #     label = 'Light'
    
    
    # norm = mpl.colors.Normalize(vmin=-0.4, vmax=0)
    color = my_cmap(spectrum.voltage * -2.5)
    

    
    if spectrum.voltage < 0.1:
        # spectrum.y = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1500, fs = 60000)
        ax.plot(spectrum.x, spectrum.y - spectrum.y.min(), label = spectrum.voltage, color = color)
        print(-spectrum.voltage * 10)
    
ax.set_xlim(40, 180)
ax.set_ylim(-0.001, 0.25)
# ax.set_yscale('symlog')
# ax.legend(title = 'Potential v. Ag/AgCl (V)', ncol = 3)
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# ax.legend(by_label.values(), by_label.keys(), title = 'Potential v. Ag/AgCl (V)', ncol = 2)

ylim = ax.get_ylim()
bars = np.arange(0, int(spectrum.x.max()), 60)
ax.bar(bars, 1, width = 30, align = 'edge', zorder = 1, color = 'grey', alpha = 0.5, label = 'Dark')
ax.bar(bars + 30, 1, width = 30, align = 'edge', zorder = 1, color = 'white', alpha = 1, label = 'Light')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), title = 'Potential v. Ag/AgCl (V)', ncol = 2, facecolor = 'lightgrey')

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + '30s Toggle Photocurrent CA' + '.svg', format = 'svg')
    plt.close(fig)
    


#%% Plot dark/light current separation v. potential from 10s toggle spectra

particle = my_h5['Potentiostat']
# particle = Particle()
particle_name = 'Co-TAPP-SMe 60nm MLAgg on ITO'


# 10s Light toggle photocurrent plots at 0.0V to -0.4V

## Get all spectra to plot with times
keys = list(particle.keys())
keys = natsort.natsorted(keys)
spectra = []
timestamps = []
for key in keys:    
    if 'light_toggle_10s' in key and key != 'light_toggle_10s_CA_0.0V_20' and key != 'light_toggle_10s_CA_0.0V_0' and key != 'light_toggle_10s_CA_-0.3V_0' and key != 'light_toggle_10s_CA_-0.4V_1':
        spectrum = particle[key]
        timestamp = spectrum.attrs['creation_timestamp']
        voltage = spectrum.attrs['Levels_v (V)']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = key
        spectrum.voltage = voltage[0]
        spectrum.truncate(40, 180)
        spectra.append(spectrum)
        
## Sort spectra by time
# spectra = natsort.natsorted(spectra, key = lambda x : x.timestamp)

## Plot dark & LSV in time order
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Potential v. Ag/AgCl(V)')
ax.set_ylabel('Photocurrent ($\mu$A)')
ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Light Toggle 10s')
offset = 0
my_cmap = plt.get_cmap('viridis')
for i, spectrum in enumerate(spectra):
    # if 'dark' in spectrum.name:
    #     color = 'grey'
    #     label = 'Dark'
    # else:
    #     color = 'orange'
    #     label = 'Light'
    
    
    # norm = mpl.colors.Normalize(vmin=-0.4, vmax=0)
    color = my_cmap(spectrum.voltage * -2.5)
    
  
    if spectrum.voltage < 0.1:
        spectrum.y = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1500, fs = 60000)
        
        
        for i in range(0, 7):
            current_dark = spectrum.y
        
        ## Plot current at 180s - current at 150s
        current_180 = np.mean(spectrum.y[1390:1399])
        current_150 = np.mean(spectrum.y[1090:1100])
        ax.scatter(spectrum.voltage, current_180 - current_150, label = spectrum.voltage, color = color, s = 100)
        
        ## Plot current at 120s - current at 90s
        current_120 = np.mean(spectrum.y[790:800])
        current_90 = np.mean(spectrum.y[490:500])
        ax.scatter(spectrum.voltage, current_120 - current_90, label = spectrum.voltage, color = color, s = 100)

save = False
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + '10s Photocurrent' + '.svg', format = 'svg')
    plt.close(fig)
    
#%% Plot dark/light current separation v. potential from 30s toggle spectra

particle = my_h5['Potentiostat']
# particle = Particle()
particle_name = 'Co-TAPP-SMe 60nm MLAgg on ITO'


# 10s Light toggle photocurrent plots at 0.0V to -0.4V

## Get all spectra to plot with times
keys = list(particle.keys())
keys = natsort.natsorted(keys)
spectra = []
timestamps = []
for key in keys:    
    if 'light_toggle_30s' in key:# and key != 'light_toggle_30s_CA_-0.4V_1' and key != 'light_toggle_30s_CA_-0.3V_0': #and key != 'light_toggle_10s_CA_0.0V_0' and key != 'light_toggle_10s_CA_-0.3V_0' and key != 'light_toggle_10s_CA_-0.4V_1':
        spectrum = particle[key]
        timestamp = spectrum.attrs['creation_timestamp']
        voltage = spectrum.attrs['Levels_v (V)']
        spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
        spectrum.timestamp = timestamp[timestamp.find('T')+1:]
        spectrum.name = key
        spectrum.voltage = voltage[0]
        spectrum.truncate(40, 180)
        spectra.append(spectrum)
        
## Sort spectra by time
# spectra = natsort.natsorted(spectra, key = lambda x : x.timestamp)

## Plot dark & LSV in time order
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Potential v. Ag/AgCl(V)')
ax.set_ylabel('Photocurrent ($\mu$A)')
ax.set_title('Co-TAPP-SMe MLAgg Photocurrent - Light Toggle 30s')
offset = 0
my_cmap = plt.get_cmap('viridis')
for i, spectrum in enumerate(spectra):
    # if 'dark' in spectrum.name:
    #     color = 'grey'
    #     label = 'Dark'
    # else:
    #     color = 'orange'
    #     label = 'Light'
    
    
    # norm = mpl.colors.Normalize(vmin=-0.4, vmax=0)
    color = my_cmap(spectrum.voltage * -2.5)
    
  
    if spectrum.voltage < 0.1:
        spectrum.y = spt.butter_lowpass_filt_filt(spectrum.y, cutoff = 1500, fs = 60000)
        
        ## Plot current at 180s - current at 150s
        current_180 = np.mean(spectrum.y[1390:1399])
        current_150 = np.mean(spectrum.y[1090:1100])
        ax.scatter(spectrum.voltage, current_180 - current_150, label = spectrum.voltage, color = color, s = 100)
        
        ## Plot current at 120s - current at 90s
        current_120 = np.mean(spectrum.y[790:800])
        current_90 = np.mean(spectrum.y[490:500])
        ax.scatter(spectrum.voltage, current_120 - current_90, label = spectrum.voltage, color = color, s = 100)

save = True
if save == True:
    save_dir = get_directory(particle.name)
    fig.savefig(save_dir + particle_name + '30s Photocurrent' + '.svg', format = 'svg')
    plt.close(fig)
#%%

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
    