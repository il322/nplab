# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Co-TAPP-SMe SERS from Renishaw h5 file
(adapted from 2024-01-05_633nm_Kinetic_Decay_Plotter_v2.py)

    - Compares 633nm & 785nm SERS for Co-TAPP-SMe MLAgg and Bare ITO with sample face up and face down
    - Compare calibration to 2024-04-10.h5 Lab 1 Co-TAPP-SMe

Data: 
    2024-07-11_Renishaw_Co-TAPP-SMe_60nm_MLAgg_on_ITO Raman Data.h5
    2024-04-10_633nm-BPT-MLAgg-Powerseries_633nm-Co-TAPP-SMe_MLAgg-Powerswitch_CCD-Flatness.h5

(samples:
     2024-06-04_Co-TAPP-SMe_60nm_MLAgg_on_ITO_b
     2023-11-28_Co-TAPP-SMe_MLAgg_on_Glass_c)

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
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-07-11_Renishaw_Co-TAPP-SMe_60nm_MLAgg_on_ITO\2024-07-11_Renishaw_Co-TAPP-SMe_60nm_MLAgg_on_ITO Raman Data.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory():
        
    directory_path = r"C:\Users\il322\Desktop\Offline Data\2024-07-11 Renishaw Co-TAPP Analysis\\" #+ particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path

#%% Spectral calibration

# Using 2024-04-10.h5 data as comparison to Renishaw

cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-04-10_633nm-BPT-MLAgg-Powerseries_633nm-Co-TAPP-SMe_MLAgg-Powerswitch_CCD-Flatness.h5")


# Spectral calibration

## Get default literature BPT spectrum & peaks
lit_spectrum, lit_wn = cal.process_default_lit_spectrum()

## Load BPT ref spectrum
bpt_ref = cal_h5['ref_meas']['BPT_ref']
bpt_ref = SERS.SERS_Spectrum(bpt_ref)

## Coarse adjustments to miscalibrated spectra
coarse_shift = 230 # coarse shift to ref spectrum
coarse_stretch = 0.91 # coarse stretch to ref spectrum
notch_range = [(70 + coarse_shift) * coarse_stretch, (128 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
truncate_range = [notch_range[1] + 75, None] # Truncate range for all spectra on this calibration - Add 50 to take out notch slope

## Convert to wn
bpt_ref.x = spt.wl_to_wn(bpt_ref.x, 632.8)
bpt_ref.x = bpt_ref.x + coarse_shift
bpt_ref.x = bpt_ref.x * coarse_stretch

## No notch spectrum (use this truncation for all spectra!)
bpt_ref_no_notch = bpt_ref
bpt_ref_no_notch.truncate(start_x = truncate_range[0], end_x = truncate_range[1])

# Baseline, smooth, and normalize no notch ref for peak finding
bpt_ref_no_notch.y_baselined = bpt_ref_no_notch.y -  spt.baseline_als(y=bpt_ref_no_notch.y,lam=1e1,p=1e-4,niter=1000)
bpt_ref_no_notch.y_smooth = spt.butter_lowpass_filt_filt(bpt_ref_no_notch.y_baselined,
                                                        cutoff=1000,
                                                        fs = 11000,
                                                        order=2)
bpt_ref_no_notch.normalise(norm_y = bpt_ref_no_notch.y_smooth)

## Find BPT ref peaks
ref_wn = cal.find_ref_peaks(bpt_ref_no_notch, lit_spectrum = lit_spectrum, lit_wn = lit_wn, threshold = 0.05, distance = 1)

# ref_wn[3] = bpt_ref_no_notch.x[371]

## Find calibrated wavenumbers
wn_cal = cal.calibrate_spectrum(bpt_ref_no_notch, ref_wn, lit_spectrum = lit_spectrum, lit_wn = lit_wn, linewidth = 1, deg = 2)
bpt_ref.x = wn_cal


#%% Spectral efficiency white light calibration

white_ref = cal_h5['ref_meas']['white_scatt_x5']
white_ref = SERS.SERS_Spectrum(white_ref.attrs['wavelengths'], white_ref[2], title = 'White Scatterer')

## Convert to wn
white_ref.x = spt.wl_to_wn(white_ref.x, 632.8)
white_ref.x = white_ref.x + coarse_shift
white_ref.x = white_ref.x * coarse_stretch

## Get white bkg (counts in notch region)
#notch = SERS.SERS_Spectrum(white_ref.x[np.where(white_ref.x < (notch_range[1]-50))], white_ref.y[np.where(white_ref.x < (notch_range[1] - 50))], name = 'White Scatterer Notch') 
# notch = SERS.SERS_Spectrum(x = spt.truncate_spectrum(white_ref.x, white_ref.y, notch_range[0], notch_range[1] - 100)[0], 
#                             y = spt.truncate_spectrum(white_ref.x, white_ref.y, notch_range[0], notch_range[1] - 100)[1], 
#                             name = 'White Scatterer Notch')
# notch = SERS.SERS_Spectrum(white_ref.x, white_ref.y, title = 'Notch')
# notch_range = [(70 + coarse_shift) * coarse_stretch, (105 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
# notch.truncate(notch_range[0], notch_range[1])
# notch_cts = notch.y.mean()
# notch.plot(title = 'White Scatter Notch')

# ## Truncate out notch (same as BPT ref), assign wn_cal
white_ref.truncate(start_x = truncate_range[0], end_x = truncate_range[1])


## Convert back to wl for efficiency calibration
white_ref.x = spt.wn_to_wl(white_ref.x, 632.8)


# Calculate R_setup

R_setup = cal.white_scatter_calibration(wl = white_ref.x,
                                    white_scatter = white_ref.y,
                                    white_bkg = 330,
                                    plot = True,
                                    start_notch = None,
                                    end_notch = None,
                                    bpt_ref = bpt_ref)

## Get dark counts - skip for now as using powerseries
# dark_cts = my_h5['PT_lab']['whire_ref_x5']
# dark_cts = SERS.SERS_Spectrum(wn_cal_633, dark_cts[5], title = 'Dark Counts')
# # dark_cts.plot()
# plt.show()

''' 
Still issue with 'white background' of calculating R_setup
Right now, choosing white background ~400 (near notch counts) causes R_setup to be very low at long wavelengths (>900nm)
This causes very large background past 1560cm-1 BPT peak
Using a white_bkg of -100000 flattens it out...
'''    


#%% Plot Lab 1 Co-TAPP-SMe

## 2024-04-10 Co-TAPP-SMe spectrum
spectrum = cal_h5['ParticleScannerScan_3']['Particle_2']['SERS_Powerseries_0']
spectrum = SERS.SERS_Spectrum(spectrum)
spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
spectrum.x = spectrum.x + coarse_shift
spectrum.x = spectrum.x * coarse_stretch
spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
spectrum.x = wn_cal
# spectrum.calibrate_intensity(R_setup, dark_counts = 300)
spectrum.normalise()
spectrum04 = spectrum

## Plot both
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.plot(spectrum04.x, spectrum04.y_norm, color = 'red', label = 'Co-TAPP 04-10')
ax.legend()


#%%

## Laser power % to actual (measured) laser power in uW
dict_633 = {0.1 : 1.5, 5 : 15, 100 : 1730}
dict_785 = {0.0001 : 3.35, 1 : 1900, 10 : 12000, 50 : 25000}

group = my_h5['All Raw']
spectra = []

good_spectra = ['Spectrum 14',
                'Spectrum 15',
                'Spectrum 16',
                'Spectrum 17',
                'Spectrum 28',
                'Spectrum 29',
                'Spectrum 30',
                'Spectrum 31',
                'Spectrum 32',
                'Spectrum 34',
                'Spectrum 36',
                'Spectrum 37',
                'Spectrum 38',
                'Spectrum 39',
                'Spectrum 40',
                'Spectrum 42',
                'Spectrum 43',
                'Spectrum 00',
                'Spectrum 01',
                'Spectrum 04',
                'Spectrum 05',
                'Spectrum 06',
                'Spectrum 07',
                'Spectrum 08',
                'Spectrum 09'
                ]

for spectrum in list(group.keys()):
    
    if spectrum[:spectrum.find(':')] not in good_spectra:
        continue
    
    spectrum = group[spectrum]
    spectrum = SERS.SERS_Spectrum(spectrum)
    
    ## Get correct power & int time
    spectrum.integration_time = (spectrum.integration_time / 1000) * spectrum.accumulations
    spectrum.laser_power = spectrum.laser_power[spectrum.laser_power.find(' ')+1:spectrum.laser_power.rfind('%')]
    if spectrum.laser_wavelength == 633:
        spectrum.laser_power = dict_633[float(spectrum.laser_power)]
    elif spectrum.laser_wavelength == 785:
        try:
            spectrum.laser_power = dict_785[float(spectrum.laser_power)]
        except:
            spectrum.laser_power = 0
        
    spectrum.y = spectrum.y/(spectrum.laser_power * spectrum.integration_time)
    
    spectra.append(spectrum)
    
#%% Plot 785nm MLAgg & Bare ITO Face Up & Face Down

fig, ax = plt.subplots(1,1,figsize=[16,9])
ax.set_title('785 nm SERS - Co-TAPP-SMe 60 nm MLAgg on ITO', fontsize = 'x-large')
ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
ax.set_ylabel('SERS Intensity (cts/$\mu$W/s)', fontsize = 'large')

lines = []

y_max = 0

for spectrum in spectra:
    
    if spectrum.laser_wavelength != 785:
        continue

    if 'FaceUp' in spectrum.name:
        linestyle = '-'
    elif 'FaceDown' in spectrum.name:
        linestyle = ':'

    
    if 'Co-TAPP' in spectrum.name:
        color = 'orange'
        label = 'Co-TAPP-SMe MLAgg'
        if spectrum.laser_power != 3.35:
            continue
    else:
        color = 'sienna'
        label = 'Bare ITO'
        
    ax.plot(spectrum.x, spectrum.y, color = color, linestyle = linestyle)
    
    if spectrum.y.max() > y_max:
        y_max = spectrum.y.max()
    
ax.plot(spectrum04.x, (spectrum04.y/spectrum04.y.max()) * y_max, color = 'blue', zorder = 1, alpha = 0.3)
    
## Legend stuff
ax.plot([],[], color = 'orange', linestyle = '-', linewidth = 15, label = 'Co-TAPP-SMe MLAgg')
ax.plot([],[], color = 'sienna', linestyle = '-', linewidth = 15, label = 'Bare ITO')
ax.plot([],[], color = 'blue', alpha = 0.3, linestyle = '-', linewidth = 15, label = 'Lab 1 Co-TAPP-SMe')
ax.plot([],[], color = 'black', linestyle = '-', label = 'Face Up')
ax.plot([],[], color = 'black', linestyle = '--', label = 'Face Down')
ax.legend(fontsize = 'large')

plt.tight_layout(pad = 20)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + '785nm SERS with comparison' + '.svg', format = 'svg')
    plt.close(fig)


#%% Plot 633nm MLAgg & Bare ITO Face Up & Face Down

fig, ax = plt.subplots(1,1,figsize=[16,9])
ax.set_title('633 nm SERS - Co-TAPP-SMe 60 nm MLAgg on ITO', fontsize = 'x-large')
ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
ax.set_ylabel('SERS Intensity (cts/$\mu$W/s)', fontsize = 'large')

lines = []

y_max = 0

for spectrum in spectra:
    
    if spectrum.laser_wavelength != 633:
        continue

    if 'FaceUp' in spectrum.name:
        linestyle = '-'
    elif 'FaceDown' in spectrum.name:
        linestyle = ':'

    
    if 'Co-TAPP' in spectrum.name:
        color = 'red'
        label = 'Co-TAPP-SMe MLAgg'

    else:
        color = 'salmon'
        label = 'Bare ITO'
        # continue
        
    ax.plot(spectrum.x, spectrum.y, color = color, linestyle = linestyle)
    
    if spectrum.y.max() > y_max:
        y_max = spectrum.y.max()
    
ax.plot(spectrum04.x, (spectrum04.y/spectrum04.y.max()) * y_max, color = 'blue', zorder = 1, alpha = 0.3)
    
## Legend stuff
ax.plot([],[], color = 'red', linestyle = '-', linewidth = 15, label = 'Co-TAPP-SMe MLAgg')
ax.plot([],[], color = 'salmon', linestyle = '-', linewidth = 15, label = 'Bare ITO')
ax.plot([],[], color = 'blue', alpha = 0.3, linestyle = '-', linewidth = 15, label = 'Lab 1 Co-TAPP-SMe')
ax.plot([],[], color = 'black', linestyle = '-', label = 'Face Up')
ax.plot([],[], color = 'black', linestyle = '--', label = 'Face Down')
ax.legend(fontsize = 'large')

ax.set_xlim(183.59833984375, 2464.34306640625)

plt.tight_layout(pad = 20)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + '633nm SERS with comparison' + '.svg', format = 'svg')
    plt.close(fig)
    

#%% Plot all together

fig, ax = plt.subplots(1,1,figsize=[16,9])
ax.set_title('SERS - Co-TAPP-SMe 60 nm MLAgg on ITO', fontsize = 'x-large')
ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
ax.set_ylabel('SERS Intensity (cts/$\mu$W/s)', fontsize = 'large')

lines = []

y_max = 0

for spectrum in spectra:
    
    if spectrum.laser_wavelength == 633:

        if 'FaceUp' in spectrum.name:
            linestyle = '-'
        elif 'FaceDown' in spectrum.name:
            linestyle = ':'
    
        
        if 'Co-TAPP' in spectrum.name:
            color = 'red'
            label = 'Co-TAPP-SMe MLAgg'
    
        else:
            color = 'salmon'
            label = 'Bare ITO'
            # continue
            
        ax.plot(spectrum.x, spectrum.y, color = color, linestyle = linestyle)
        
    else:
        
        if 'FaceUp' in spectrum.name:
            linestyle = '-'
        elif 'FaceDown' in spectrum.name:
            linestyle = ':'

        
        if 'Co-TAPP' in spectrum.name:
            color = 'orange'
            label = 'Co-TAPP-SMe MLAgg'
            if spectrum.laser_power != 3.35:
                continue
        else:
            color = 'sienna'
            label = 'Bare ITO'
            
        ax.plot(spectrum.x, spectrum.y, color = color, linestyle = linestyle)

    if spectrum.y.max() > y_max:
        y_max = spectrum.y.max()
        
ax.plot(spectrum04.x, (spectrum04.y/spectrum04.y.max()) * y_max, color = 'blue', zorder = 1, alpha = 0.3)    

## Legend stuff
ax.plot([],[], color = 'red', linestyle = '-', linewidth = 15, label = '633 nm MLAgg')
ax.plot([],[], color = 'salmon', linestyle = '-', linewidth = 15, label = '633 nm Bare ITO')
ax.plot([],[], color = 'orange', linestyle = '-', linewidth = 15, label = '785 nm MLAgg')
ax.plot([],[], color = 'sienna', linestyle = '-', linewidth = 15, label = '785 nm Bare ITO')
ax.plot([],[], color = 'blue', alpha = 0.3, linestyle = '-', linewidth = 15, label = 'Lab 1 Co-TAPP-SMe')
ax.plot([],[], color = 'black', linestyle = '-', label = 'Face Up')
ax.plot([],[], color = 'black', linestyle = '--', label = 'Face Down')
ax.legend(fontsize = 'large')

ax.set_xlim(183.59833984375, 2464.34306640625)
# ax.set_xlim(1100, 1800)

plt.tight_layout(pad = 20)

## Save
save = False
if save == True:
    save_dir = get_directory()
    fig.savefig(save_dir + 'SERS Zoom with comparison' + '.svg', format = 'svg')
    plt.close(fig)
    
