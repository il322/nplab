# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for some photocurrent measurements with DF spectra overlay

Copied from 2024-07-15_Photocurrent_Plotter.py

Data: 2024-08-09_Co-TAPP-SMe_20nm_MLAgg_Photocurrent.h5
    


(samples: 2024-08-05_Co-TAPP-SMe_20nm_MLAgg_on_FTO_a)

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
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-08-09_Co-TAPP-SMe_20nm_MLAgg_Photocurrent.h5")

## Calibration h5 File
cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\calibration.h5")

## h5 File for saving processed data
save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2024-08_09_Processed_20nm_Photocurrent_Data.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory():
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-08-09 20nm Co-TAPP-SMe MLAgg Photocurrent Analysis\\"
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% fit equations

def cottrell(x, a, b, c):
    
    return (a/(sqrt(pi*x + c))) + b

def quadratic(x, a, b, c):
    
    return a*x**2 + b*x + c


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

group = my_h5['20nm_MLAgg_0']
# spectrum = group['Co-TAPP-SMe_600nm_50nmFWHM_toggle_30s_CA_0.0V_0']
spectrum = group['Co-TAPP-SMe_20nm_MLAgg_500nm_50nmFWHM_toggle_30s_CA_0.0V_0']
timestamp = spectrum.attrs['creation_timestamp']
# voltage = spectrum.attrs['Levels_v (V)']
voltage = 0
spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
spectrum.timestamp = timestamp[timestamp.find('T')+1:]
# spectrum.name = key
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


#%% Get list of spectra

group = my_h5['20nm_MLAgg_0']
spectra = []
skip_spectra = [
                "Co-TAPP-SMe_20nm_MLAgg_450nm_50nmFWHM_toggle_30s_CA_-0.4V_0",
                "Co-TAPP-SMe_20nm_MLAgg_650nm_50nmFWHM_toggle_30s_CA_-0.2V_0",
                "Co-TAPP-SMe_20nm_MLAgg_700nm_50nmFWHM_toggle_30s_CA_-0.2V_0",
                "Co-TAPP-SMe_20nm_MLAgg_600nm_50nmFWHM_toggle_30s_CA_0.0V_0"
                ]

for key in group.keys():
    
    if key not in skip_spectra:
        
            spectrum = group[key]
            timestamp = spectrum.attrs['creation_timestamp']
            # voltage = spectrum.attrs['Levels_v (V)']
            voltage = float(key[key.find('CA_')+3:key.find('V')])
            spectrum = SERS.SERS_Spectrum(x = np.array(spectrum)[0], y = np.array(spectrum)[1] * 10**6)
            spectrum.timestamp = timestamp[timestamp.find('T')+1:]
            spectrum.name = key
        
        
            # Baseline, smooth
            spectrum.truncate(50, None)    
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

            # spectrum.truncate(100, None)

            spectrum.y_baseline = cottrell(spectrum.x, *popt)
            spectrum.y_baselined = spectrum.y - spectrum.y_baseline
            
            spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 3000, fs = 60000)
                        
            ## Attribute handling
            spectrum.wavelength = int(key[key.find('MLAgg_')+6:key.find('MLAgg_')+9])
            spectrum.fwhm = 50
            spectrum.voltage = voltage
            spectrum.sample = 'MLAgg'
            spectrum.toggle = 100
            
            
            # Calculate PEC (light on - light off) at each toggle point of CA curve
            
            toggles = int(spectrum.x.max()/(spectrum.toggle*2))
            spectrum.toggles = toggles
            pec = []
            for i in range(0, toggles):
                dark_index = np.argmin(np.abs(spectrum.x - (100 * (2*i + 1))))
                light_index = np.argmin(np.abs(spectrum.x - (200 * (i + 1))))
                light_current = np.mean(spectrum.y_smooth[light_index -100 : light_index])
                dark_current = np.mean(spectrum.y_smooth[dark_index -100 : dark_index])
                pec.append(light_current - dark_current)
            spectrum.pec = pec
            
            spectra.append(spectrum)
            
            
## Sort spectra by voltage (for color mapping)
spectra = natsort.natsorted(spectra, key = lambda x : x.voltage)

#%% Pull in darkfield scattering and avg over MLAgg map

df_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-08-12_DF_Map_Co-TAPP-SMe_20nm_MLAgg_FTO.h5")

group = df_h5['20nm_MLAgg_map_0']

df_spectra = []

for key in list(group.keys()):
    
    if 'spec' in key:

        spectrum = df_h5['20nm_MLAgg_map_0'][key]
        spectrum = df.DF_Spectrum(spectrum)
        df_spectra.append(spectrum)
  
df_avg = []
for spectrum in df_spectra:
    
    spectrum.y = (spectrum.y - spectrum.background)/(spectrum.reference - spectrum.background)
    spectrum.truncate(400, 900)
    df_avg.append(spectrum.y)

df_avg = df.DF_Spectrum(spectrum.x, np.mean(df_avg, axis = 0), smooth_first = False)

#%% Plot light toggle CA at each wavelength

voltages = [-0.8, -0.6, -0.4, -0.2, 0.0]

fig, axes = plt.subplots(len(voltages), 2, figsize=[16,16], width_ratios = (2,1))

my_cmap = plt.get_cmap('nipy_spectral')
cmap = my_cmap
norm = mpl.colors.Normalize(vmin=420, vmax=830)
offset = 0.0


# Plot Light Toggle CA on left

for j, voltage in enumerate(voltages):

    ax = axes[j][0]

    ax.set_title(str(voltage) + ' V', fontsize = 'x-large')
    offset = 0.0
    
    if j < len(axes)-1:
        ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
            
        if spectrum.voltage != voltage:
            continue
        
        color = cmap(norm(spectrum.wavelength))  
        ax.plot(spectrum.x, (spectrum.y_smooth + i * offset), label = spectrum.voltage, color = color, linestyle = linetype, alpha = 0.7, zorder = i+2)
    
    ## Light on/off bars
    ax.set_xlim(100, 1000)
    ylim = ax.get_ylim()    
    ax.bar(np.arange((spectrum.toggle/2), spectrum.x.max() + (spectrum.toggle/2), spectrum.toggle*2), height = ylim[1] - ylim[0], bottom = ylim[0], width = 100, color = 'grey', alpha = 0.2, zorder = 0)
    ax.text(s = 'On', x = spectrum.toggle*1.1, y = ylim[0] + .05*(ylim[1]-ylim[0]), fontsize = 'large')
    ax.text(s = 'Off', x = spectrum.toggle*2.1, y = ylim[0] + .05*(ylim[1]-ylim[0]), fontsize = 'large')
    ax.set_ylim(ylim)
    
    
# Plot PEC on right

for j, voltage in enumerate(voltages):

    ax = axes[j][1]

    ax.set_title(str(voltage) + ' V', fontsize = 'x-large')
    offset = 0.0
    
    if j < len(axes)-1:
        ax.xaxis.set_ticklabels([])
    
    for i, spectrum in enumerate(spectra):
        
        linetype = 'solid'
            
        if spectrum.voltage != voltage:
            continue
        
        ## Plot
        color = cmap(norm(spectrum.wavelength))  
        ax.scatter(spectrum.wavelength, np.abs(np.mean(spectrum.pec)/power_dict[spectrum.wavelength]), color = color, s = 150)
        # ax.errorbar(spectrum.wavelength,  np.abs(np.mean(spectrum.pec)/power_dict[spectrum.wavelength]), yerr = np.abs(np.std(spectrum.pec)/(np.sqrt(len(spectrum.pec)*power_dict[spectrum.wavelength]))), marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)

        
    ylim = ax.get_ylim()
    ax.yaxis.tick_right()
    ax.plot(df_avg.x, ylim[0] + (df_avg.y - df_avg.y.min())*(ylim[1] - ylim[0])/(df_avg.y.max() - df_avg.y.min() + ylim[0]), color = 'black', alpha = 0.4, zorder = 0)
    ax.set_ylim(ylim)    
    ax.set_xlim(400, 900)
    ax.legend()
    
axes[-1][0].set_xlabel('Time (s)', fontsize = 'x-large')
axes[int((len(axes) - 1)/2)][0].set_ylabel('Current ($\mu$A)', fontsize = 'x-large')
# fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([-0.03, 0.15, 0.05, 0.7])
fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
cbar_ax.set_ylabel('Centre Wavelength (nm) - 50 nm FWHM', rotation=90, fontsize = 'x-large', labelpad = 20)
cbar_ax.yaxis.tick_left()
cbar_ax.yaxis.set_label_position("left")
fig.suptitle('Co-TAPP-SMe 20 nm MLAgg Photocurrent', fontsize = 'xx-large', horizontalalignment='center', x = 0.45, y = 0.94)


axes[-1][1].set_xlabel('Wavelength (nm)', fontsize = 'x-large')
axes[int((len(axes) - 1)/2)][1].set_ylabel('|Photocurrent| ($\mu$A/mW)', fontsize = 'x-large', rotation = 270, labelpad = 40)
axes[int((len(axes) - 1)/2)][1].yaxis.set_label_position("right")
# fig.subplots_adjust(right=0.9)


## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'Co-TAPP-SMe 20 nm MLAgg Photocurrent' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)
    
    