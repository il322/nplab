# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for some photocurrent measurements with power normalization and comparison to Bare ITO

Copied from 2024-06-28_Photocurrent_Plotter.py

Data: 2024-07-15_Co-TAPP-SME_MLAgg_Photocurrent_NKT.h5


(samples: 2024-06-04_Co-TAPP-SMe_60nm_MLAgg_on_ITO_b
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
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-07-15_Co-TAPP-SME_MLAgg_Photocurrent_NKT.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory():
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-07-15 Co-TAPP-SMe MLAgg Photocurrent Analysis\\"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% fit equations

def cottrell(x, a, b, c):
    
    return a/(sqrt(pi*x + c)) + b

def quadratic(x, a, b, c):
    
    return a*x**2 + b*x + c


#%% 100nm FWHM measurements

#%% Get power calibration as dict

group = my_h5['power_calibration_100nmFWHM_1']

power_dict = {} 

for key in group.keys():

    power = group[key]
    wavelength = int(power.attrs['wavelength'])
    power = float(np.array(power))
    new_dict = {wavelength : power}
    power_dict.update(new_dict)

#%% Testing background subtraction

# Cottrell equation background subtraction

group = my_h5['Potentiostat']
spectrum = group['Co-TAPP-SMe_650nm_toggle_30s_CA_0.20000000000000007V_0']
# spectrum = group['Bare_ITO_700nm_100nmFWHM_toggle_30s_CA_0.20000000000000007V_1']
timestamp = spectrum.attrs['creation_timestamp']
voltage = spectrum.attrs['Levels_v (V)']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
spectrum.timestamp = timestamp[timestamp.find('T')+1:]
spectrum.name = key
# spectrum.truncate(20,180)

x = spectrum.x
y = spectrum.y

try:
    popt, pcov = curve_fit(f = cottrell, 
                    xdata = x,
                    ydata = y)
        
except:
    print('Fit Error')
    # popt = [15.5,  spectrum.y[-1] - ((spectrum.y[0] - spectrum.y[-1]) * 6), 262]
    popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]
    # popt = [0.001,  spectrum.y[-1] - ((spectrum.y[0] - spectrum.y[-1]) * 1), 262]

fit_y = cottrell(spectrum.x, *popt)

plt.plot(spectrum.x, spectrum.y, color = 'black')
plt.plot(spectrum.x, fit_y, color = 'red')
# plt.plot(spectrum.x, spectrum.y - fit_y, color = 'blue')
# plt.xlim(20, 180)
# plt.ylim(-2.4, -2)


#%% Get list of spectra for 100nm FWHM Co-TAPP-SMe 30s toggle CA

group = my_h5['Potentiostat']
spectra = []
skip_spectra = [
                'Co-TAPP-SMe_500nm_toggle_10s_CA_-0.4V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_1',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_2',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.2V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_4',
                'Co-TAPP-SMe_850nm_toggle_30s_CA_-0.2V_0'
                ]

for key in group.keys():
    
    if 'toggle' in key and 'FWHM' not in key and key not in skip_spectra:
        
            spectrum = group[key]
            timestamp = spectrum.attrs['creation_timestamp']
            voltage = spectrum.attrs['Levels_v (V)']
            spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
            spectrum.timestamp = timestamp[timestamp.find('T')+1:]
            spectrum.name = key
            
            # Truncate, baseline, & smooth
            
            spectrum.truncate(20, 180)
            
            ## als baseline subtraction
            # spectrum.y_baseline = spt.baseline_als(spectrum.y, 1e7, 1e-4)
            # spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            ## Cottrell fit baseline subtraction
            x = spectrum.x
            y = spectrum.y
            try:
                popt, pcov = curve_fit(f = cottrell, 
                                    xdata = x,
                                    ydata = y)
                           
            except:
                print('\nFit Error')
                print('\n'+spectrum.name+'\n')
                popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]

            
            
            spectrum.y_baseline = cottrell(spectrum.x, *popt)
            spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 3000, fs = 60000)
            
            ## Attribute handling
            spectrum.wavelength = int(key[key.find('nm')-3:key.find('nm')])
            spectrum.fwhm = 100
            spectrum.voltage = np.round(voltage[0],2)
            if 'Co-TAPP' in key:
                spectrum.sample = 'MLAgg'
            elif 'ITO' in key:
                spectrum.sample = 'ITO'
            spectra.append(spectrum)
            
            
## Sort spectra by voltage (for color mapping)
spectra = natsort.natsorted(spectra, key = lambda x : x.voltage)

mlagg_100_spectra = spectra

#%% Get list of spectra for 100nm FWHM Bare ITO 30s toggle CA

group = my_h5['Potentiostat']

spectra = []

skip_spectra = [
                # 'Co-TAPP-SMe_500nm_toggle_10s_CA_-0.4V_0',
                # 'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_0',
                # 'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_1',
                # 'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_2'
                ]

for key in group.keys():
    
    if 'toggle' in key and '100nmFWHM' in key and key not in skip_spectra and key[-1] != '0':
        
            spectrum = group[key]
            timestamp = spectrum.attrs['creation_timestamp']
            voltage = spectrum.attrs['Levels_v (V)']
            spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
            spectrum.timestamp = timestamp[timestamp.find('T')+1:]
            spectrum.name = key
            
            # Truncate, baseline, & smooth
            
            spectrum.truncate(20, 180)
            
            ## als baseline subtraction
            # spectrum.y_baseline = spt.baseline_als(spectrum.y, 1e7, 1e-4)
            # spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            ## Cottrell fit baseline subtraction
            x = spectrum.x
            y = spectrum.y
            try:
                popt, pcov = curve_fit(f = cottrell, 
                                    xdata = x,
                                    ydata = y)
                           
            except:
                print('\nFit Error')
                print('\n'+spectrum.name+'\n')
                popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]

            
            
            spectrum.y_baseline = cottrell(spectrum.x, *popt)
            spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 3000, fs = 60000)
            
            ## Attribute handling
            spectrum.wavelength = int(key[key.find('nm')-3:key.find('nm')])
            spectrum.fwhm = 100
            spectrum.voltage = np.round(voltage[0],2)
            if 'Co-TAPP' in key:
                spectrum.sample = 'MLAgg'
            elif 'ITO' in key:
                spectrum.sample = 'ITO'
            spectra.append(spectrum)
            
            
## Sort spectra by voltage (for color mapping)
spectra = natsort.natsorted(spectra, key = lambda x : x.voltage)

ito_100_spectra = spectra


#%% Plot 30s light toggle CA at each wavelength

spectra = mlagg_100_spectra + ito_100_spectra

voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

fig, axes = plt.subplots(5,1,figsize=[12,16])

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
offset = 0.0

for j, voltage in enumerate(voltages):

    ax = axes[j]

    ax.set_title(str(voltage) + ' V', x = 0.85, y = 0.8)
    offset = 0.0


    
    if j < len(axes)-1:
        ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        if spectrum.sample == 'ITO':
            continue
            linetype = 'solid'
        else:
            # continue
            linetype = 'solid'    
            
        if spectrum.voltage != voltage:
            continue
        
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x, (spectrum.y_smooth + i * offset), label = spectrum.voltage, color = color, linestyle = linetype)
        
    
axes[-1].set_xlabel('Time (s)', fontsize = 'x-large')
axes[int((len(axes) - 1)/2)].set_ylabel('Current ($\mu$A)', fontsize = 'x-large')
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar_ax.set_ylabel('Centre Wavelength (nm)', rotation=270, fontsize = 'x-large', labelpad = 40)
fig.suptitle('Co-TAPP-SMe MLAgg Photocurrent\nLight Toggle 30s - 100nm FWHM', fontsize = 'x-large', horizontalalignment='center', x = 0.56, y = 0.95)


## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'MLAgg 30s Toggle Photocurrent 100nm FWHM' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)


#%% Calculate photocurrent difference for each spectrum and plot

spectra = mlagg_100_spectra + ito_100_spectra

voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

fig, axes = plt.subplots(5,1,figsize=[11,17], sharex = False, sharey = False)

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
offset = 0.0

for j, voltage in enumerate(voltages):

    ax = axes[j]

    ax.set_title(str(voltage) + ' V', x = 0.9, y = 0.8)
    offset = 0.0
    
    if j < len(axes)-1:
        ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        if spectrum.sample == 'ITO':
            continue
            linetype = 'solid'
        else:
            # continue
            linetype = 'solid'    
            
        if spectrum.voltage != voltage:
            continue
        
        ## Calculate PEC at 3 points (180s - 150s; 120s - 90s; 60s - 30s)
        pec1 = np.mean(spectrum.y_smooth[390:400]) - np.mean(spectrum.y_smooth[90:100])
        pec2 = np.mean(spectrum.y_smooth[990:1000]) - np.mean(spectrum.y_smooth[690:700])
        pec3 = np.mean(spectrum.y_smooth[1590:1600]) - np.mean(spectrum.y_smooth[1290:1300])
        
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        ax.scatter(spectrum.wavelength, (pec1), label = spectrum.voltage, color = color, s = 150)
        ax.scatter(spectrum.wavelength, (pec2), label = spectrum.voltage, color = color, s = 150)
        ax.scatter(spectrum.wavelength, (pec3), label = spectrum.voltage, color = color, s = 150)
        ax.hlines(y = 0, xmin = 430, xmax = 870, color = 'grey', zorder = 0, linestyle = 'dashed')
        ax.set_xlim(430.0, 870.0)

    
axes[-1].set_xlabel('Centre Wavelength (nm)', fontsize = 'x-large', labelpad = 10)
axes[int((len(axes) - 1)/2)].set_ylabel('Current ($\mu$A)', fontsize = 'x-large', labelpad = 10)
fig.suptitle('ITO Photocurrent\n100nm FWHM', fontsize = 'x-large', horizontalalignment='center')
plt.tight_layout()

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'ITO PEC 100nm FWHM' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)


#%% Calculate photocurrent difference for each spectrum and plot - POWER NORMALIZED

spectra = mlagg_100_spectra + ito_100_spectra

voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

fig, axes = plt.subplots(5,1,figsize=[11,17], sharex = False, sharey = False)

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
offset = 0.0

for j, voltage in enumerate(voltages):

    ax = axes[j]

    ax.set_title(str(voltage) + ' V', x = 0.9, y = 0.8)
    offset = 0.0
    
    if j < len(axes)-1:
        ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        if spectrum.sample == 'ITO':
            # continue
            linetype = 'solid'
        else:
            continue
            linetype = 'solid'    
            
        if spectrum.voltage != voltage:
            continue
        
        ## Calculate PEC at 3 points (180s - 150s; 120s - 90s; 60s - 30s)
        pec1 = np.mean(spectrum.y_smooth[390:400]) - np.mean(spectrum.y_smooth[90:100])
        pec2 = np.mean(spectrum.y_smooth[990:1000]) - np.mean(spectrum.y_smooth[690:700])
        pec3 = np.mean(spectrum.y_smooth[1590:1600]) - np.mean(spectrum.y_smooth[1290:1300])
        
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        ax.scatter(spectrum.wavelength, (pec1)/power_dict[spectrum.wavelength], label = spectrum.voltage, color = color, s = 150)
        ax.scatter(spectrum.wavelength, (pec2)/power_dict[spectrum.wavelength], label = spectrum.voltage, color = color, s = 150)
        ax.scatter(spectrum.wavelength, (pec3)/power_dict[spectrum.wavelength], label = spectrum.voltage, color = color, s = 150)
        ax.hlines(y = 0, xmin = 430, xmax = 870, color = 'grey', zorder = 0, linestyle = 'dashed')
        ax.set_xlim(430.0, 870.0)

    
axes[-1].set_xlabel('Centre Wavelength (nm)', fontsize = 'x-large', labelpad = 10)
axes[int((len(axes) - 1)/2)].set_ylabel('Current ($\mu$A/mW)', fontsize = 'x-large', labelpad = 10)
fig.suptitle('ITO Photocurrent\nPower Normalized - 100nm FWHM', fontsize = 'x-large', horizontalalignment='center')
plt.tight_layout()

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'ITO PEC 100nm FWHM Power Normalized' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)    
    
    
#%% 50nm FWHM Measurements


#%% Get power calibration as dict

group = my_h5['power_calibration_50nmFWHM_0']

power_dict = {} 

for key in group.keys():

    power = group[key]
    wavelength = int(power.attrs['wavelength'])
    power = float(np.array(power))
    new_dict = {wavelength : power}
    power_dict.update(new_dict)


#%% Testing background subtraction

# Cottrell equation background subtraction

group = my_h5['Potentiostat']
# spectrum = group['Co-TAPP-SMe_600nm_50nmFWHM_toggle_30s_CA_0.0V_0']
spectrum = group['Bare_ITO_500nm_50nmFWHM_toggle_30s_CA_0.0V_1']
timestamp = spectrum.attrs['creation_timestamp']
voltage = spectrum.attrs['Levels_v (V)']
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
spectrum.timestamp = timestamp[timestamp.find('T')+1:]
spectrum.name = key
spectrum.truncate(20,180)

x = spectrum.x
y = spectrum.y

try:
    popt, pcov = curve_fit(f = cottrell, 
                    xdata = x,
                    ydata = y)
        
except:
    print('Fit Error')
    # popt = [15.5,  spectrum.y[-1] - ((spectrum.y[0] - spectrum.y[-1]) * 6), 262]
    popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]
    # popt = [0.001,  spectrum.y[-1] - ((spectrum.y[0] - spectrum.y[-1]) * 1), 262]

fit_y = cottrell(spectrum.x, *popt)

plt.plot(spectrum.x, spectrum.y, color = 'black')
plt.plot(spectrum.x, fit_y, color = 'red')
# plt.plot(spectrum.x, spectrum.y - fit_y, color = 'blue')
# plt.xlim(20, 180)
# plt.ylim(-2.4, -2)


#%% Get list of spectra for 100nm FWHM Co-TAPP-SMe 30s toggle CA

group = my_h5['Potentiostat']
spectra = []
skip_spectra = [
                'Co-TAPP-SMe_500nm_toggle_10s_CA_-0.4V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_1',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_2',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.2V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_4',
                'Co-TAPP-SMe_850nm_toggle_30s_CA_-0.2V_0',
                'Co-TAPP-SMe_Dark_CA_-0.4V_0',
                'Co-TAPP-SMe_Dark_CA_-0.4V_1'
                ]

for key in group.keys():
    
    if 'Co-TAPP-SMe' in key and '50nmFWHM' in key and key not in skip_spectra:
        
            spectrum = group[key]
            timestamp = spectrum.attrs['creation_timestamp']
            voltage = spectrum.attrs['Levels_v (V)']
            spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
            spectrum.timestamp = timestamp[timestamp.find('T')+1:]
            spectrum.name = key
            
            # Truncate, baseline, & smooth
            
            spectrum.truncate(20, 180)
            
            ## als baseline subtraction
            # spectrum.y_baseline = spt.baseline_als(spectrum.y, 1e7, 1e-4)
            # spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            ## Cottrell fit baseline subtraction
            x = spectrum.x
            y = spectrum.y
            try:
                popt, pcov = curve_fit(f = cottrell, 
                                    xdata = x,
                                    ydata = y)
                           
            except:
                print('\nFit Error')
                print('\n'+spectrum.name+'\n')
                popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]

            
            
            spectrum.y_baseline = cottrell(spectrum.x, *popt)
            spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 3000, fs = 60000)
            
            ## Attribute handling
            spectrum.wavelength = int(key[key.find('nm')-3:key.find('nm')])
            spectrum.fwhm = 50
            spectrum.voltage = np.round(voltage[0],2)
            if 'Co-TAPP' in key:
                spectrum.sample = 'MLAgg'
            elif 'ITO' in key:
                spectrum.sample = 'ITO'
            spectra.append(spectrum)
            
            
## Sort spectra by voltage (for color mapping)
spectra = natsort.natsorted(spectra, key = lambda x : x.voltage)

mlagg_50_spectra = spectra

#%% Get list of spectra for 100nm FWHM Co-TAPP-SMe 30s toggle CA

group = my_h5['Potentiostat']
spectra = []
skip_spectra = [
                'Co-TAPP-SMe_500nm_toggle_10s_CA_-0.4V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_1',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_2',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.2V_0',
                'Co-TAPP-SMe_500nm_toggle_30s_CA_-0.4V_4',
                'Co-TAPP-SMe_850nm_toggle_30s_CA_-0.2V_0',
                'Co-TAPP-SMe_Dark_CA_-0.4V_0',
                'Co-TAPP-SMe_Dark_CA_-0.4V_1',
                'Bare_ITO_450nm_50nmFWHM_toggle_30s_CA_-0.4V_1'
                ]

for key in group.keys():
    
    if 'ITO' in key and '50nmFWHM' in key and key[-1] != str(0) and key not in skip_spectra:
        
            spectrum = group[key]
            timestamp = spectrum.attrs['creation_timestamp']
            voltage = spectrum.attrs['Levels_v (V)']
            spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
            spectrum.timestamp = timestamp[timestamp.find('T')+1:]
            spectrum.name = key
            
            # Truncate, baseline, & smooth
            
            spectrum.truncate(20, 180)
            
            ## als baseline subtraction
            # spectrum.y_baseline = spt.baseline_als(spectrum.y, 1e7, 1e-4)
            # spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            ## Cottrell fit baseline subtraction
            x = spectrum.x
            y = spectrum.y
            try:
                popt, pcov = curve_fit(f = cottrell, 
                                    xdata = x,
                                    ydata = y)
                           
            except:
                print('\nFit Error')
                print('\n'+spectrum.name+'\n')
                popt = [(spectrum.y[0] - spectrum.y[1300])/0.026,  spectrum.y[1300] * 0.985, np.abs(108/(spectrum.y[0] - spectrum.y[1300]))]

            
            
            spectrum.y_baseline = cottrell(spectrum.x, *popt)
            spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 3000, fs = 60000)
            
            ## Attribute handling
            spectrum.wavelength = int(key[key.find('nm')-3:key.find('nm')])
            spectrum.fwhm = 50
            spectrum.voltage = np.round(voltage[0],2)
            if 'Co-TAPP' in key:
                spectrum.sample = 'MLAgg'
            elif 'ITO' in key:
                spectrum.sample = 'ITO'
            spectra.append(spectrum)
            
            
## Sort spectra by voltage (for color mapping)
spectra = natsort.natsorted(spectra, key = lambda x : x.voltage)

ito_50_spectra = spectra


#%% Plot 30s light toggle CA at each wavelength

spectra = mlagg_50_spectra + ito_50_spectra

voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

fig, axes = plt.subplots(5,1,figsize=[12,16])

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
offset = 0.0

for j, voltage in enumerate(voltages):

    ax = axes[j]

    ax.set_title(str(voltage) + ' V', x = 0.85, y = 0.8)
    offset = 0.0


    
    if j < len(axes)-1:
        ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        if spectrum.sample == 'ITO':
            # continue
            linetype = 'solid'
        else:
            continue
            linetype = 'solid'    
            
        if spectrum.voltage != voltage:
            continue
        
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x, (spectrum.y_smooth + i * offset), label = spectrum.voltage, color = color, linestyle = linetype)
        
    
axes[-1].set_xlabel('Time (s)', fontsize = 'x-large')
axes[int((len(axes) - 1)/2)].set_ylabel('Current ($\mu$A)', fontsize = 'x-large')
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar_ax.set_ylabel('Centre Wavelength (nm)', rotation=270, fontsize = 'x-large', labelpad = 40)
fig.suptitle('ITO Photocurrent\nLight Toggle 30s - 50nm FWHM', fontsize = 'x-large', horizontalalignment='center', x = 0.56, y = 0.95)


## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'ITO 30s Toggle Photocurrent 50nm FWHM' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)


#%% Calculate photocurrent difference for each spectrum and plot

spectra = mlagg_50_spectra + ito_50_spectra

voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

fig, axes = plt.subplots(5,1,figsize=[11,17], sharex = False, sharey = False)

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
offset = 0.0

for j, voltage in enumerate(voltages):

    ax = axes[j]

    ax.set_title(str(voltage) + ' V', x = 0.9, y = 0.8)
    offset = 0.0
    
    if j < len(axes)-1:
        ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        if spectrum.sample == 'ITO':
            continue
            linetype = 'solid'
        else:
            # continue
            linetype = 'solid'    
            
        if spectrum.voltage != voltage:
            continue
        
        ## Calculate PEC at 3 points (180s - 150s; 120s - 90s; 60s - 30s)
        pec1 = np.mean(spectrum.y_smooth[390:400]) - np.mean(spectrum.y_smooth[90:100])
        pec2 = np.mean(spectrum.y_smooth[990:1000]) - np.mean(spectrum.y_smooth[690:700])
        pec3 = np.mean(spectrum.y_smooth[1590:1600]) - np.mean(spectrum.y_smooth[1290:1300])
        
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        ax.scatter(spectrum.wavelength, (pec1), label = spectrum.voltage, color = color, s = 150)
        ax.scatter(spectrum.wavelength, (pec2), label = spectrum.voltage, color = color, s = 150)
        ax.scatter(spectrum.wavelength, (pec3), label = spectrum.voltage, color = color, s = 150)
        ax.hlines(y = 0, xmin = 430, xmax = 870, color = 'grey', zorder = 0, linestyle = 'dashed')
        ax.set_xlim(430.0, 870.0)

    
axes[-1].set_xlabel('Centre Wavelength (nm)', fontsize = 'x-large', labelpad = 10)
axes[int((len(axes) - 1)/2)].set_ylabel('Current ($\mu$A)', fontsize = 'x-large', labelpad = 10)
fig.suptitle('Co-TAPP-SMe MLAgg Photocurrent\n50nm FWHM', fontsize = 'x-large', horizontalalignment='center')
plt.tight_layout()

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'MLAgg PEC 50nm FWHM' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)


#%% Calculate photocurrent difference for each spectrum and plot - POWER NORMALIZED

spectra = mlagg_50_spectra + ito_50_spectra

voltages = [-0.4, -0.2, 0.0, 0.2, 0.4]

fig, axes = plt.subplots(5,1,figsize=[11,17], sharex = False, sharey = False)

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
offset = 0.0

for j, voltage in enumerate(voltages):

    ax = axes[j]

    ax.set_title(str(voltage) + ' V', x = 0.9, y = 0.8)
    offset = 0.0
    
    if j < len(axes)-1:
        ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        if spectrum.sample == 'ITO':
            continue
            linetype = 'solid'
        else:
            # continue
            linetype = 'solid'    
            
        if spectrum.voltage != voltage:
            continue
        
        ## Calculate PEC at 3 points (180s - 150s; 120s - 90s; 60s - 30s)
        pec1 = np.mean(spectrum.y_smooth[390:400]) - np.mean(spectrum.y_smooth[90:100])
        pec2 = np.mean(spectrum.y_smooth[990:1000]) - np.mean(spectrum.y_smooth[690:700])
        pec3 = np.mean(spectrum.y_smooth[1590:1600]) - np.mean(spectrum.y_smooth[1290:1300])
        
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        ax.scatter(spectrum.wavelength, (pec1)/power_dict[spectrum.wavelength], label = spectrum.voltage, color = color, s = 150)
        ax.scatter(spectrum.wavelength, (pec2)/power_dict[spectrum.wavelength], label = spectrum.voltage, color = color, s = 150)
        ax.scatter(spectrum.wavelength, (pec3)/power_dict[spectrum.wavelength], label = spectrum.voltage, color = color, s = 150)
        ax.hlines(y = 0, xmin = 430, xmax = 870, color = 'grey', zorder = 0, linestyle = 'dashed')
        ax.set_xlim(430.0, 870.0)

    
axes[-1].set_xlabel('Centre Wavelength (nm)', fontsize = 'x-large', labelpad = 10)
axes[int((len(axes) - 1)/2)].set_ylabel('Current ($\mu$A/mW)', fontsize = 'x-large', labelpad = 10)
fig.suptitle('Co-TAPP-SMe MLAgg Photocurrent\nPower Normalized - 50nm FWHM', fontsize = 'x-large', horizontalalignment='center')
plt.tight_layout()

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'MLAgg PEC 50nm FWHM Power Normalized' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)    
    
    