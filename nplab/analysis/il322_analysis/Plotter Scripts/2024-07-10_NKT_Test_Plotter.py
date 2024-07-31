# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 23:42:43 2024

@author: il322

Plotter for NKT wavelength & power test


"""

import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
import tkinter as tk
from tkinter import filedialog
import statistics
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter
from pylab import *
import nplab
import h5py
import natsort
import os

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

#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name = None):
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-07-10 NKT Test Analysis\\"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% Load h5

my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-07-10_NKT_Test.h5")


#%% Plot spectra of each set wavelength

group = my_h5['varia_spectrum_10nm_width_free_space_2']


mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[16,10])
fig.suptitle('NKT Varia Monochromator Test - 10nm FWHM', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', fontsize = 'large')
ax.set_ylabel('Normalized Intensity (a.u.)', fontsize = 'large')
offset = 0.0

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)

for i, spectrum in enumerate(list(group.keys())):
     
    # if 'Bentham' not in spectrum:
    #     continue
    
    name = spectrum[0:3]
    
    wln = int(name)
    color = my_cmap(norm(wln))
    spectrum = group[spectrum]
    background = np.array(spectrum.attrs['background'])
    spectrum = spt.Spectrum(spectrum)
    spectrum.y = spectrum.y - background
    
    # if spectrum.y.max() < 400:
    #     continue
    
    if wln < 380:
        continue
    
    spectrum.normalise()
    ax.plot(spectrum.x, spectrum.y, color = color, label = name, linewidth = 2)
    print(name)

plt.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap), ax = ax, label = 'Set Wavelength (nm)')    
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
ax.set_xlim(390,870)
# ax.set_ylim(0, 1.3)
plt.tight_layout(pad = 0.5)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT 10nm Spectra' + '.svg', format = 'svg')
    plt.close(fig)
    

#%% Plot power of each set wavelength - 10nm FWHM

group = my_h5['varia_power_10nm_width_free_space_1']


mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[12,8])
fig.suptitle('NKT Varia Monochromator Test - 10nm FWHM', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', fontsize = 'large')
ax.set_ylabel('Power (mW)', fontsize = 'large')
offset = 0.0

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
wlns = []
powers = []

for i, spectrum in enumerate(list(group.keys())):
     
    # if 'Bentham' not in spectrum:
    #     continue
    
    name = spectrum[0:3]
    
    wln = int(name)
    color = my_cmap(norm(wln))
    spectrum = group[spectrum]
    power = np.array(spectrum)
    wlns.append(wln)
    powers.append(power)
    # background = np.array(spectrum.attrs['background'])
    # spectrum = spt.Spectrum(spectrum)
    # spectrum.y = spectrum.y - background
    
    # if spectrum.y.max() < 400:
    #     continue
    
    if wln < 380:
        continue
    
    # spectrum.normalise()
    ax.scatter(wln, power, color = color, label = name, linewidth = 2, zorder = 2, s = 100)
    # print(name)
ax.plot(wlns, powers, zorder = 1, color = 'black', linewidth = 1)
# plt.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap), ax = ax, label = 'Set Wavelength (nm)')    
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
ax.set_xlim(390,870)
# ax.set_ylim(0, 1.3)
plt.tight_layout(pad = 0.5)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT 10nm Power' + '.svg', format = 'svg')
    plt.close(fig)
    

#%% Plot power of each set wavelength - 20nm FWHM

group = my_h5['varia_power_20nm_width_free_space_0']


mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[12,8])
fig.suptitle('NKT Varia Monochromator Test - 20nm FWHM', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', fontsize = 'large')
ax.set_ylabel('Power (mW)', fontsize = 'large')
offset = 0.0

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
wlns = []
powers = []

for i, spectrum in enumerate(list(group.keys())):
     
    # if 'Bentham' not in spectrum:
    #     continue
    
    name = spectrum[0:3]
    
    wln = int(name)
    color = my_cmap(norm(wln))
    spectrum = group[spectrum]
    power = np.array(spectrum)
    wlns.append(wln)
    powers.append(power)
    # background = np.array(spectrum.attrs['background'])
    # spectrum = spt.Spectrum(spectrum)
    # spectrum.y = spectrum.y - background
    
    # if spectrum.y.max() < 400:
    #     continue
    
    if wln < 380:
        continue
    
    # spectrum.normalise()
    ax.scatter(wln, power, color = color, label = name, linewidth = 2, zorder = 2, s = 100)
    # print(name)
ax.plot(wlns, powers, zorder = 1, color = 'black', linewidth = 1)
# plt.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap), ax = ax, label = 'Set Wavelength (nm)')    
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
ax.set_xlim(390,870)
# ax.set_ylim(0, 1.3)
plt.tight_layout(pad = 0.5)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT 20nm Power' + '.svg', format = 'svg')
    plt.close(fig)


#%% Plot set v actual wavelength - 10nm FWHM

group = my_h5['varia_spectrum_10nm_width_free_space_2']


mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[16,10])
fig.suptitle('NKT Varia Monochromator Test - 10nm FWHM', fontsize = 'x-large')
ax.set_xlabel('Set Wavelength (nm)', fontsize = 'large')
ax.set_ylabel('Actual Wavelength (nm)', fontsize = 'large')
offset = 0.0

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)

for i, spectrum in enumerate(list(group.keys())):
     
    # if 'Bentham' not in spectrum:
    #     continue
    
    name = spectrum[0:3]
    
    wln = int(name)
    color = my_cmap(norm(wln))
    spectrum = group[spectrum]
    background = np.array(spectrum.attrs['background'])
    spectrum = spt.Spectrum(spectrum)
    spectrum.y = spectrum.y - background
    
    # if spectrum.y.max() < 400:
    #     continue
    
    # if spectrum.y.max() < 400:
    #     continue
    
    # if wln < 300:
    #     continue
    
    spectrum.normalise()
    spectrum.y_norm_smooth = spt.butter_lowpass_filt_filt(spectrum.y_norm, cutoff = 3000, fs = 40000, order = 2)
    maxima = spt.detect_maxima(spectrum.y_norm_smooth, lower_threshold=0.2)
    
    for maximum in maxima:
        size = 200 * spectrum.y_norm[maximum] 
        ax.scatter(wln, spectrum.x[maximum], color = color, s = size, zorder = 2, marker = 'o')

ax.plot(np.linspace(400,850,100), np.linspace(400,850,100), color = 'black', zorder = 1)

# plt.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap), ax = ax, label = 'Set Wavelength (nm)')    
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
ax.set_xlim(390,870)
# ax.set_ylim(0, 1.3)
plt.tight_layout(pad = 0.5)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT 10nm Set v Actual Wavelength' + '.svg', format = 'svg')
    plt.close(fig)


#%% Plot FWHM of each wavelength - 10nm FWHM

group = my_h5['varia_spectrum_10nm_width_free_space_2']


mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[16,10])
fig.suptitle('NKT Varia Monochromator Test - 10nm FWHM', fontsize = 'x-large')
ax.set_xlabel('Set Wavelength (nm)', fontsize = 'large')
ax.set_ylabel('FWHM (nm)', fontsize = 'large')
offset = 0.0

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)

for i, spectrum in enumerate(list(group.keys())):
     
    # if 'Bentham' not in spectrum:
    #     continue
    
    name = spectrum[0:3]
    
    wln = int(name)
    color = my_cmap(norm(wln))
    spectrum = group[spectrum]
    background = np.array(spectrum.attrs['background'])
    spectrum = spt.Spectrum(spectrum)
    spectrum.y = spectrum.y - background
    
    # if spectrum.y.max() < 400:
    #     continue
    
    # if spectrum.y.max() < 400:
    #     continue
    
    # if wln < 300:
    #     continue

    spectrum.normalise()
    peak_wlns = spectrum.x[np.where(spectrum.y_norm >= 0.5)]
    fwhm = peak_wlns.max() - peak_wlns.min()
    ax.scatter(wln, fwhm, color = color, s = 200, zorder = 2, marker = 'o')

ax.hlines(10, 400, 860, color = 'black')
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
ax.set_xlim(390,870)
# ax.set_ylim(0, 1.3)
plt.tight_layout(pad = 0.5)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT 10nm FWHM' + '.svg', format = 'svg')
    plt.close(fig)


#%% Plot spectra v varying bandwidth

group = my_h5['varia_spectrum_varying_width_free_space_0']


mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[16,10])
fig.suptitle('NKT Varia Monochromator Test - Varying Bandwidth', fontsize = 'x-large')
ax.set_xlabel('Wavelength (nm)', fontsize = 'large')
ax.set_ylabel('Intensity (a.u.)', fontsize = 'large')
offset = 0.0

my_cmap = plt.get_cmap('inferno')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=-100, vmax=500)

bandwidths = np.geomspace(1,400,50)

for i, spectrum in enumerate(natsort.natsorted(list(group.keys()))):
     
    
    spectrum = group[spectrum]
    bandwidth = bandwidths[i]
    
    print(bandwidth)
    color = my_cmap(norm(bandwidth))
    background = np.array(spectrum.attrs['background'])
    spectrum = spt.Spectrum(spectrum)
    spectrum.y = spectrum.y - background
    
    spectrum.normalise()
    ax.plot(spectrum.x, spectrum.y, color = color, label = name, linewidth = 2)
    # print(name)

plt.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap), ax = ax, label = 'Set Bandwidth (nm)')    
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
ax.set_xlim(390,870)
# ax.set_ylim(0, 1.3)
plt.tight_layout(pad = 0.5)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT Varying Bandwidth Spectra' + '.svg', format = 'svg')
    plt.close(fig)
    
    
#%% Plot FWHM v varying bandwidth

group = my_h5['varia_spectrum_varying_width_free_space_0']


mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[16,10])
fig.suptitle('NKT Varia Monochromator Test - Varying Bandwidth', fontsize = 'x-large')
ax.set_xlabel('Set Bandwidth (nm)', fontsize = 'large')
ax.set_ylabel('Actual FWHM (nm)', fontsize = 'large')
offset = 0.0

# my_cmap = plt.get_cmap('inferno')
# cmap = my_cmap
# norm = mpl.colors.Normalize(vmin=-100, vmax=500)
bandwidths = np.geomspace(1,400,50)

for i, spectrum in enumerate(natsort.natsorted(list(group.keys()))):
     
    
    spectrum = group[spectrum]
    bandwidth = bandwidths[i]
    color = 'black'
    background = np.array(spectrum.attrs['background'])
    spectrum = spt.Spectrum(spectrum)
    spectrum.y = spectrum.y - background
    
    spectrum.normalise()
    peak_wlns = spectrum.x[np.where(spectrum.y_norm >= 0.5)]
    fwhm = peak_wlns.max() - peak_wlns.min()
    ax.scatter(bandwidth, fwhm, color = color, linewidth = 2, s = 70)
    # print(name)
ax.plot(np.linspace(0,100, 100), np.linspace(0,100,100))
# plt.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap), ax = ax, label = 'Set Bandwidth (nm)')    
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
# ax.set_xlim(0,20)
# ax.set_ylim(0, 20)
plt.tight_layout(pad = 0.5)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT Varying Bandwidth FWHM' + '.svg', format = 'svg')
    plt.close(fig)
    
#%% Plot power stability over time

mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[16,10])
fig.suptitle('NKT Varia Power Stability Test - 10nm FWHM', fontsize = 'x-large')
ax.set_xlabel('Time (s)', fontsize = 'large')
ax.set_ylabel('$\Delta$Power (mW)', fontsize = 'large')
offset = 0.0

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)


for key in my_h5.keys():
    if'power_stability' in key:
        
        group = my_h5[key]
        
        wln = int(key[22:25])
        
        
                
        for i, spectrum in enumerate(list(group.keys())):
             
            # if 'Bentham' not in spectrum:
            #     continue
        
            color = my_cmap(norm(wln))
            spectrum = group[spectrum]
            power = np.array(spectrum)
            if i == 0:
                power0 = power
            # wlns.append(wln)
            # powers.append(power)
            
            time = spectrum.attrs['time']             

            

            ax.scatter(time, power - power0, color = color, label = wln, linewidth = 2, zorder = 2, s = 100)
            # print(name)
            
    
# ax.plot(wlns, powers, zorder = 1, color = 'black', linewidth = 1)
# plt.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap), ax = ax, label = 'Set Wavelength (nm)')    
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
# ax.set_xlim(390,870)
# ax.set_ylim(0, 1.3)
plt.tight_layout(pad = 0.5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
leg1 = ax.legend((by_label.values()), (by_label.keys()), title = 'Centre Wln (nm)', ncol = 1, loc = 'upper right')
ax.add_artist(leg1)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT 10nm Power Stability' + '.svg', format = 'svg')
    plt.close(fig)

#%% Plot power stability over time as % change

mpl.rcParams['lines.linewidth'] = 0.2
plt.rc('font', size=18, family='sans-serif')
plt.rc('lines', linewidth=3)
fig, (ax) = plt.subplots(1, 1, figsize=[16,10])
fig.suptitle('NKT Varia Power Stability Test - 10nm FWHM', fontsize = 'x-large')
ax.set_xlabel('Time (s)', fontsize = 'large')
ax.set_ylabel('% Change in Power', fontsize = 'large')
offset = 0.0

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)


for key in my_h5.keys():
    if'power_stability' in key:
        
        group = my_h5[key]
        
        wln = int(key[22:25])
        
        
                
        for i, spectrum in enumerate(list(group.keys())):
             
            # if 'Bentham' not in spectrum:
            #     continue
        
            color = my_cmap(norm(wln))
            spectrum = group[spectrum]
            power = np.array(spectrum)
            if i == 0:
                power0 = power
            # wlns.append(wln)
            # powers.append(power)
            
            time = spectrum.attrs['time']             

            

            ax.scatter(time, ((power - power0)/power0)*100, color = color, label = wln, linewidth = 2, zorder = 2, s = 100)
            # print(name)
            
    
# ax.plot(wlns, powers, zorder = 1, color = 'black', linewidth = 1)
# plt.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=my_cmap), ax = ax, label = 'Set Wavelength (nm)')    
# ax.legend(loc = 'upper center', fontsize = 'medium', ncol = 8, title = 'Set Wavelength (nm)')
# ax.set_xlim(390,870)
# ax.set_ylim(0, 1.3)
plt.tight_layout(pad = 0.5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
leg1 = ax.legend((by_label.values()), (by_label.keys()), title = 'Centre Wln (nm)', ncol = 1, loc = 'upper right')
ax.add_artist(leg1)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'NKT 10nm Power Stability Percent Change' + '.svg', format = 'svg')
    plt.close(fig)
