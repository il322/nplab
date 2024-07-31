# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:32:45 2023

@author: il322

v2 of 2023-12-01_Powerseries_powerswitch_Plotter

Just adds baseline subtraction and gets single powerseries I needed for a presentation


(samples:
     2023-11-28_Co-TAPP-SMe_60nm_MLAgg_on_Glass_b)

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

#%% Load h5


my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2023-12-01_MLAgg_633nm_Powerseries_Powerswitch.h5")


#%%

# Spectral calibration

## Get default literature BPT spectrum & peaks
lit_spectrum, lit_wn = cal.process_default_lit_spectrum()

## Load BPT ref spectrum
bpt_ref = my_h5['ref_meas']['BPT_633nm']
bpt_ref = SERS.SERS_Spectrum(bpt_ref)

## Coarse adjustments to miscalibrated spectra
coarse_shift = 50 # coarse shift to ref spectrum
coarse_stretch = 1 # coarse stretch to ref spectrum
notch_range = [(127 + coarse_shift) * coarse_stretch, (177 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
truncate_range = [notch_range[1] + 50] # Truncate range for all spectra on this calibration - Add 50 to take out notch slope

## Convert to wn
bpt_ref.x = spt.wl_to_wn(bpt_ref.x, 632.8)
bpt_ref.x = bpt_ref.x + coarse_shift
bpt_ref.x = bpt_ref.x * coarse_stretch

## No notch spectrum (use this truncation for all spectra!)
bpt_ref_no_notch = bpt_ref
bpt_ref_no_notch.truncate(start_x = truncate_range[0], end_x = None)

# Baseline, smooth, and normalize no notch ref for peak finding
bpt_ref_no_notch.y_baselined = bpt_ref_no_notch.y -  spt.baseline_als(y=bpt_ref_no_notch.y,lam=1e1,p=1e-4,niter=1000)
bpt_ref_no_notch.y_smooth = spt.butter_lowpass_filt_filt(bpt_ref_no_notch.y_baselined,
                                                        cutoff=2000,
                                                        fs = 10000,
                                                        order=2)
bpt_ref_no_notch.normalise(norm_y = bpt_ref_no_notch.y_smooth)

## Find BPT ref peaks
ref_wn = cal.find_ref_peaks(bpt_ref_no_notch, lit_spectrum = lit_spectrum, lit_wn = lit_wn, threshold = 0.06, distance = 20)

## Find calibrated wavenumbers
ref_wn2 = [458.83089403163467, 530.6365633210862, 725.27638512, 1070.441854368437, 1131.2076337917642, 1335.996877754811, 1638.3042298284472]
wn_cal = cal.calibrate_spectrum(bpt_ref_no_notch, ref_wn2, lit_spectrum = lit_spectrum, lit_wn = lit_wn, linewidth = 1, deg = 2)
bpt_ref.x = wn_cal


#%% Spectral efficiency white light calibration

white_ref = my_h5['ref_meas']['white_ref_x5']
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
notch = SERS.SERS_Spectrum(white_ref.x, white_ref.y, title = 'Notch')
notch.truncate(notch_range[0], notch_range[1])
notch_cts = notch.y.mean()
notch.plot(title = 'White Scatter Notch')

# ## Truncate out notch (same as BPT ref), assign wn_cal
white_ref.truncate(start_x = truncate_range[0], end_x = None)


## Convert back to wl for efficiency calibration
white_ref.x = spt.wn_to_wl(white_ref.x, 632.8)


# Calculate R_setup

R_setup = cal.white_scatter_calibration(wl = white_ref.x,
                                    white_scatter = white_ref.y,
                                    white_bkg = notch_cts,
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


#%% Powerseries plotting


#%% Dark counts power series - MLAgg

particle = my_h5['dark_powerseries_2']


# Add all SERS spectra to powerseries list in order

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'powerseries' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.x = spectrum.x + coarse_shift
    spectrum.truncate(start_x = truncate_range[0], end_x = None)
    spectrum.x = wn_cal
    spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 2, cutoff = 200)
    powerseries[i] = spectrum
    
dark_powerseries = powerseries


# List of powers used, for colormaps

powers_list = []
colors_list = np.linspace(0,10,10)

for spectrum in dark_powerseries:
    powers_list.append(spectrum.laser_power)
    

# Add jump back to min powers to dark powerseries

dark_powerseries = np.insert(dark_powerseries, np.linspace(2,len(dark_powerseries),9).astype(int), dark_powerseries[0])
for spec in dark_powerseries:
    print(spec.laser_power)


#%% Plot single direct powerseries for single MLAgg spot

particle = my_h5['ParticleScannerScan_0']['Particle_0']


# Add all SERS spectra to powerseries list in order

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []

for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])


fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')

for i, spectrum in enumerate(powerseries[0:15]):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.x = spectrum.x + coarse_shift
    spectrum.truncate(start_x = truncate_range[0], end_x = None)
    spectrum.x = wn_cal
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y,
                                  exposure = spectrum.cycle_time)
    spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    spectrum.y_baselined = spectrum.y - spectrum.baseline
    
    #spectrum.y = spt.remove_cosmic_rays(spectrum.y)
    #spectrum.truncate(1100, 1700)
    #spectrum.y_baselined = spt.baseline_als(spectrum.y, 1e0, 1e-1, niter = 10)
    #baseline = np.polyfit(spectrum.x, spectrum.y, 1)
    #spectrum.y_baselined = spectrum.y - (spectrum.x * baseline[0] + baseline[1])
    #spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined,cutoff = 3000, fs = 20000, order = 2)
    #spectrum.normalise(norm_y = spectrum.y)
    
    ## Color = color of previous power in powerseries
    my_cmap = plt.get_cmap('inferno')
    color = my_cmap(i/15)
    if i == 0:
        spectrum.plot(ax = ax, plot_y = spectrum.y_baselined, title = '633nm SERS Powerseries - Co-TAPP-SME AuNP MLAgg', linewidth = 2, color = color, label = 1.0, zorder = 30-i)
    elif spectrum.laser_power >= 0.0012:
        spectrum.plot(ax = ax, plot_y = spectrum.y_baselined, title = '633nm SERS Powerseries - Co-TAPP-SME AuNP MLAgg', linewidth = 2, color = color, label = np.round(dark_powerseries[i].laser_power * 1000,0), zorder = 30-i)

    ## Labeling & plotting
    ax.legend(fontsize = 18, ncol = 5, loc = 'upper center')
    ax.get_legend().set_title('Laser power ($\mu$W)')
    for line in ax.get_legend().get_lines():
        line.set_linewidth(4.0)
    # fig.suptitle('MLAgg')
    powerseries[i] = spectrum
    
    ax.set_xlim(1200, 1700)
    ax.set_ylim(-5000, 100000)
    plt.tight_layout(pad = 2)
    



