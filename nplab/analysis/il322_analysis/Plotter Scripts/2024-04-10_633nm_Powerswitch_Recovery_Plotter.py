# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Co-TAPP-SMe 633nm SERS Powerswitch & Recovery

Data: 2024-04-10_633nm-BPT-MLAgg-Powerseries_633nm-Co-TAPP-SMe_MLAgg-Powerswitch_CCD-Flatness.h5


(samples:
     2023-11-28_Co-TAPP-SMe_60nm_MLAgg_c)

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


#%%


class Particle(): 
    def __init__(self):
        self.name = None

class Peak():
    def __init__(self, mu, height, width, sigma, area, **kwargs):
        self.height = height
        self.width = width
        self.sigma = sigma
        self.area = area
        self.mu = mu


#%% h5 files

## Load raw data h5
my_h5 = h5py.File(r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-04-10_633nm-BPT-MLAgg-Powerseries_633nm-Co-TAPP-SMe_MLAgg-Powerswitch_CCD-Flatness.h5")


#%% Spectral calibration

# Spectral calibration

## Get default literature BPT spectrum & peaks
lit_spectrum, lit_wn = cal.process_default_lit_spectrum()

## Load BPT ref spectrum
bpt_ref = my_h5['ref_meas']['BPT_ref']
bpt_ref = SERS.SERS_Spectrum(bpt_ref)

## Coarse adjustments to miscalibrated spectra
coarse_shift = 230 # coarse shift to ref spectrum
coarse_stretch = 0.91 # coarse stretch to ref spectrum
notch_range = [(70 + coarse_shift) * coarse_stretch, (128 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
truncate_range = [notch_range[1] + 50, None] # Truncate range for all spectra on this calibration - Add 50 to take out notch slope

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

white_ref = my_h5['ref_meas']['white_scatt_x5']
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


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r'C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-04-10_633nm Powerswitch Recovery Analysis\_' + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path



#%% 633nm powerswitch dark counts

particle = my_h5['ref_meas']


# Add all SERS spectra to powerseries list in order

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []
for key in keys:
    if 'dark_powerswitch' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    if i == 0:
        spectrum.cycle_time = 50
        spectrum.laser_power = 0.002
    elif i == 1:
        spectrum.cycle_time = 1.111109972000122
        spectrum.laser_power = 0.08999999999999998
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.x = spectrum.x + coarse_shift
    spectrum.x = spectrum.x * coarse_stretch
    spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    spectrum.x = wn_cal
    spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 10, cutoff = 50)
    powerseries[i] = spectrum
    
dark_powerseries = powerseries

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('Counts')

for i, spectrum in enumerate(dark_powerseries):
    my_cmap = plt.get_cmap('inferno')
    color = my_cmap(i/15)
    ax.plot(spectrum.x, spectrum.y, color = color, alpha = 0.5)
    # ax.plot(spectrum.x, spectrum.y_cosmic, linestyle = '--', color = color)
    ax.set_ylim(0,1000)


## Duplicate dark powerseries to match regular powerseries
particle = my_h5['ParticleScannerScan_3']['Particle_0']
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])
while len(dark_powerseries) < len(powerseries):
    dark_powerseries.append(dark_powerseries[0])
    dark_powerseries.append(dark_powerseries[1])
    
## List of powers used, for colormaps
# powers_list = []
# for spectrum in dark_powerseries:
    # powers_list.append(spectrum.laser_power)
    # print(spectrum.cycle_time)
    
    
# Plot dark subtracted as test

particle = my_h5['ParticleScannerScan_3']['Particle_0']

## Add all SERS spectra to powerseries list in order
keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
for key in keys:
    if 'SERS' in key:
        powerseries.append(particle[key])

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')

## Process and plot
for i, spectrum in enumerate(powerseries[2:4]):
  
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.x = spectrum.x + coarse_shift
    spectrum.x = spectrum.x * coarse_stretch
    spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    spectrum.x = wn_cal
    
    ## Plot raw, baseline, baseline subtracted
    offset = 2000
    spectrum.plot(ax = ax, plot_y = spectrum.y + (i*offset), linewidth = 1, color = 'black', label = i, zorder = 30-i)
    spectrum.plot(ax = ax, plot_y = (spectrum.y - dark_powerseries[i].y) + (i*offset), linewidth = 1, color = 'darkgreen', label = i, zorder = 30-i)
    dark_powerseries[i].plot(ax = ax, title = 'Dark Counts Subtraction Test', color = 'grey', linewidth = 1)  


#%% Testing background subtraction & cosmic ray removal


particle = my_h5['ParticleScannerScan_3']['Particle_0']


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


for i, spectrum in enumerate(powerseries[2:4]):
  
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.x = spectrum.x + coarse_shift
    spectrum.x = spectrum.x * coarse_stretch
    spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    spectrum.x = wn_cal
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y,
                                  exposure = spectrum.cycle_time)
    
    spectrum.y_cosmic = spt.remove_cosmic_rays(spectrum.y, threshold = 9)

    ## Baseline
    spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    spectrum.y_baselined = spectrum.y - spectrum.baseline
    spectrum.y_cosmic = spectrum.y_cosmic - spectrum.baseline
    
    ## Plot raw, baseline, baseline subtracted
    offset = 50000
    spectrum.plot(ax = ax, plot_y = (spectrum.y - spectrum.y.min()) + (i*offset), linewidth = 1, color = 'black', label = 'Raw', zorder = 1)
    spectrum.plot(ax = ax, plot_y = spectrum.y_baselined + (i*offset), linewidth = 1, color = 'purple', alpha = 0.5, label = 'Background subtracted', zorder = 2)
    spectrum.plot(ax = ax, plot_y = spectrum.baseline- spectrum.y.min() + (i*offset), title = 'Background Subtraction & Cosmic Ray Test', color = 'darkred', label = 'Background', linewidth = 1)    
    spectrum.plot(ax = ax, plot_y = spectrum.y_cosmic + (i*offset), title = 'Background Subtraction & Cosmic Ray Test', color = 'purple', label = 'Cosmic ray removed', linewidth = 1, linestyle = '--', zorder = 3)
    fig.suptitle(particle.name)    
    # ax.set_xlim(1200, 1700)
    # ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
    plt.tight_layout(pad = 0.8)
    
#%% Get all particles to analyze into Particle class with h5 locations and in a list


particles = []

scan_list = ['ParticleScannerScan_3']

# Loop over particles in target particle scan

for particle_scan in scan_list:
    particle_list = []
    particle_list = natsort.natsorted(list(my_h5[particle_scan].keys()))
    
    ## Loop over particles in particle scan
    for particle in particle_list:
        if 'Particle' not in particle:
            particle_list.remove(particle)
           
            
    # Loop over particles in particle scan
    
    for particle in particle_list[0:93]:
        
        ## Save to class and add to list
        this_particle = Particle()
        this_particle.name = 'MLAgg_' + str(particle_scan) + '_' + particle
        this_particle.h5_address = my_h5[particle_scan][particle]
        particles.append(this_particle)


# Make avg particle

# avg_particle = Particle()
# avg_particle.h5_address = particles[0].h5_address
# avg_particle.name = 'MLAgg_Avg'    
# particles.append(avg_particle)

#%% Functions to add & process SERS powerseries for each particle


def process_powerseries(particle):
    
    ## Add all SERS spectra to powerseries list in order
    keys = list(particle.h5_address.keys())
    keys = natsort.natsorted(keys)
    powerseries = []
    for key in keys:
        if 'SERS' in key:
            powerseries.append(particle.h5_address[key])
    
    ## Loop over SERS spectra in powerseries, process, and add back to powerseries
    for i, spectrum in enumerate(powerseries):
        
        ## x-axis truncation, calibration
        spectrum = SERS.SERS_Spectrum(spectrum)
        spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
        spectrum.x = spectrum.x + coarse_shift
        spectrum.x = spectrum.x * coarse_stretch
        spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
        spectrum.x = wn_cal
        spectrum.calibrate_intensity(R_setup = R_setup,
                                      dark_counts = dark_powerseries[i].y,
                                      exposure = spectrum.cycle_time,
                                      laser_power = spectrum.laser_power)
        
        spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
        spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
        spectrum.y_baselined = spectrum.y - spectrum.baseline
        spectrum.normalise(norm_y = spectrum.y_baselined)
        powerseries[i] = spectrum

    return powerseries


def process_powerswitch_recovery(particle):
    
    particle.dark_time = float(np.array(particle.h5_address['dark_time_0']))
    
    ## Add all SERS spectra to powerseries list in order
    keys = list(particle.h5_address.keys())
    keys = natsort.natsorted(keys)
    powerseries = []
    for key in keys:
        if 'SERS' in key:
            powerseries.append(particle.h5_address[key])
    
    ## Loop over SERS spectra in powerseries, process, and add back to powerseries
    for i, spectrum in enumerate(powerseries):
        
        ## x-axis truncation, calibration
        spectrum = SERS.SERS_Spectrum(spectrum)
        spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
        spectrum.x = spectrum.x + coarse_shift
        spectrum.x = spectrum.x * coarse_stretch
        spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
        spectrum.x = wn_cal
        spectrum.calibrate_intensity(R_setup = R_setup,
                                      dark_counts = dark_powerseries[i].y,
                                      exposure = spectrum.cycle_time,
                                      laser_power = spectrum.laser_power)
        # spectrum.truncate(None, 1800)
        spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
        spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
        spectrum.y_baselined = spectrum.y - spectrum.baseline
        # spectrum.normalise(norm_y = spectrum.y_baselined)
        powerseries[i] = spectrum
    
    particle.powerseries = powerseries
        


#%% Loop over all particles and process powerswitch

for particle in tqdm(particles, leave = True):
    
    process_powerswitch_recovery(particle)


#%%
# Calculate average powerseries of all particles

avg_particle = particles[len(particles)-1]
powerseries = avg_particle.powerseries

## Set powerseries to 0

for i, spectrum in enumerate(powerseries):
    spectrum.y = np.zeros(spectrum.y.shape)
    spectrum.baseline = np.zeros(spectrum.baseline.shape)
    spectrum.y_baselined = np.zeros(spectrum.y_baselined.shape)

## Add powerseries from all particles    
for particle in particles[0:len(particles)-2]:
    for i, spectrum in enumerate(powerseries):
        spectrum.y += particle.powerseries[i].y
        spectrum.baseline += particle.powerseries[i].baseline
        spectrum.y_baselined += particle.powerseries[i].y_baselined

## Divide to get average
for i, spectrum in enumerate(powerseries):
    spectrum.y = spectrum.y/(len(particles)-1)
    spectrum.baseline = spectrum.baseline/(len(particles)-1)
    spectrum.y_baselined = spectrum.y_baselined/(len(particles)-1)
  
#%% Plot min powerseries, direct powerseries, and timescan powerseries for each particle

    
# Plotting prep
my_cmap = plt.get_cmap('inferno')


def plot_min_powerseries(particle, save = False):
    
    
    powerseries = particle.powerseries
    powerseries_y = particle.powerseries_y
    
    fig, ax = plt.subplots(1,1,figsize=[18,12])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
    ax.set_ylabel('SERS Intensity (cts/mW/s)')
    
    ## Plot min power spectra only
    for i, spectrum in enumerate(powerseries):
        offset = 0
        color = my_cmap(i/(len(dark_powerseries)+2))
        if i == 0:
            previous_power = '0.0'
        else:
            previous_power = np.round(powerseries[i-1].laser_power * 1000, 0)
        if spectrum.laser_power <= 0.0029:
            spectrum.plot(ax = ax, plot_y = spectrum.y_baselined + (i*offset), title = '785nm Min Power Powerseries - Co-TAPP-SMe 60nm MLAgg', linewidth = 1, color = color, label = previous_power, zorder = (19-i))
    
        ## Labeling & plotting
        ax.legend(fontsize = 18, ncol = np.ceil((len(powerseries)+1)/4), loc = 'upper center')
        ax.get_legend().set_title('Previous laser power ($\mu$W)')
        for line in ax.get_legend().get_lines():
            line.set_linewidth(4.0)
        fig.suptitle(particle.name)
        ax.set_xlim(450, 1800)
        ax.set_ylim(-500, powerseries_y.max() * 1.3)
        fig.tight_layout(pad = 0.8)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '785nm Min Powerseries' + '.svg', format = 'svg')
        plt.close(fig)
    
    
def plot_direct_powerseries(particle, save = False):
    
    
    powerseries = particle.powerseries
    powerseries_y = particle.powerseries_y
    
    fig2, ax2 = plt.subplots(1,1,figsize=[18,9])
    ax2.set_xlabel('Raman Shifts (cm$^{-1}$)')
    ax2.set_ylabel('SERS Intensity (cts/mW/s)')
    
    ## Plot direct powerseries
    for i, spectrum in enumerate(powerseries):
        offset = 0
        color = my_cmap(i/(len(dark_powerseries)+2))
        if i % 2 == 0:
            spectrum.plot(ax = ax2, plot_y = spectrum.y_baselined + (i*offset), title = '785nm Direct Powerseries - Co-TAPP-SMe 60nm MLAgg', linewidth = 1, color = color, label = np.round(spectrum.laser_power * 1000, 0), zorder = (19-i))
    
        ## Labeling & plotting
        ax2.legend(fontsize = 18, ncol = 5, loc = 'upper center')
        ax2.get_legend().set_title('Laser power ($\mu$W)')
        for line in ax2.get_legend().get_lines():
            line.set_linewidth(4.0)
        fig2.suptitle(particle.name)
        ax2.set_xlim(450, 1800)
        ax2.set_ylim(-500, powerseries_y.max() * 1.3)
        fig2.tight_layout(pad = 0.8)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig2.savefig(save_dir + particle.name + '785nm Direct Powerseries' + '.svg', format = 'svg')
        plt.close(fig2)
    

def plot_timescan_powerseries(particle, save = False):
    
    
    powerseries = particle.powerseries
    powerseries_y = particle.powerseries_y

    ## Plot powerseries as timescan
    timescan = SERS.SERS_Timescan(x = spectrum.x, y = powerseries_y, exposure = 1)
    fig3, (ax3) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.arange(0,len(powerseries),1)
    v_min = powerseries_y.min()
    v_max = np.percentile(powerseries_y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax3.set_yticklabels([])
    ax3.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    ax3.set_xlim(450,1700)
    ax3.set_title('785nm Powerseries' + 's\n' + str(particle.name), fontsize = 'x-large', pad = 10)
    pcm = ax3.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig3.colorbar(pcm, ax=ax3)
    clb.set_label(label = 'SERS Intensity', size = 'large', rotation = 270, labelpad=30)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig3.savefig(save_dir + particle.name + '633nm Powerswitch Timescan' + '.svg', format = 'svg')
        plt.close(fig)
        
def plot_timescan_powerswitch_recovery(particle, save = False):
    
    
    powerseries = particle.powerseries
    
    ## Add dark times into powerseries (for plotting)
    if particle.dark_time > 0:
        spectrum = SERS.SERS_Spectrum(x = powerseries[0].x, y = np.zeros(len(powerseries[0].y)))
        spectrum.y_baselined = spectrum.y        
        powerseries = np.insert(powerseries, [10], spectrum)
        powerseries = np.insert(powerseries, [10], spectrum)
        powerseries = np.insert(powerseries, [10], spectrum)
        powerseries = np.insert(powerseries, [23], spectrum)
        powerseries = np.insert(powerseries, [23], spectrum)
        powerseries = np.insert(powerseries, [23], spectrum)
        powerseries = np.insert(powerseries, [36], spectrum)
        powerseries = np.insert(powerseries, [36], spectrum)
        powerseries = np.insert(powerseries, [36], spectrum)

    ## Get all specrta into single array for timescan
    powerseries_y = np.zeros((len(powerseries), len(powerseries[0].y)))
    for i,spectrum in enumerate(powerseries):
        powerseries_y[i] = spectrum.y_baselined
    powerseries_y = np.array(powerseries_y)

    ## Plot powerseries as timescan
    timescan = SERS.SERS_Timescan(x = spectrum.x, y = powerseries_y, exposure = 1)
    fig3, (ax3) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.arange(0,len(powerseries),1)
    v_min = powerseries_y.min()
    v_max = np.percentile(powerseries_y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax3.set_yticklabels([])
    ax3.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    # ax3.set_xlim(450,2100)
    if particle.dark_time > 0:
        ax3.text(x=820,y=10.5,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
        ax3.text(x=820,y=23.5,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
        ax3.text(x=820,y=36.5,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
    ax3.set_title('633 nm Powerswitch Recovery - 2 $\mu$W / 90 $\mu$W' + '\n' + str(particle.name), fontsize = 'x-large', pad = 10)
    pcm = ax3.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig3.colorbar(pcm, ax=ax3)
    clb.set_label(label = 'SERS Intensity', size = 'large', rotation = 270, labelpad=30)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig3.savefig(save_dir + particle.name + '633nm Powerswitch Recovery' + '.svg', format = 'svg')
        plt.close(fig3)


# Loop over all particles and plot

for particle in tqdm(particles, leave = True):
    
    # ## Get all specrta into single array for timescan
    # powerseries_y = np.zeros((len(particle.powerseries), len(particle.powerseries[0].y)))
    # for i,spectrum in enumerate(particle.powerseries):
    #     powerseries_y[i] = spectrum.y_baselined
    # particle.powerseries_y = np.array(powerseries_y)
    
    # plot_min_powerseries(particle)
    # plot_direct_powerseries(particle)
    plot_timescan_powerswitch_recovery(particle, save = True)


#%% Peak fitting functions
       

def gaussian(x, height, center, width, width_height_frac = 0.5):
    a = height
    b = center
    c = width/(2*np.sqrt(2*np.log(1/width_height_frac)))
    
    return a*np.exp(-(((x - b)**2)/(2*c**2)))
        
def lorentzian(x, height, center, fwhm):
    I = height
    x0 = center
    gamma = fwhm/2
    numerator = gamma**2
    denominator = (x - x0)**2 + gamma**2
    quot = numerator/denominator
    
    y = I*quot
    return y


#%% Fit peaks of individual regions

particle = particles[11]
powerseries = particle.powerseries


# 1620 peak

fit_range = [1351, 1460]

height_frac = 0.5

for i, spectrum in enumerate(powerseries):
    
    
    fig, ax = plt.subplots(1,1,figsize=[18,9])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
    ax.set_ylabel('SERS Intensity (cts/mW/s)')
    spectrum.plot(ax = ax, plot_y = spectrum.y_baselined, color = 'black')
    spectrum.plot(ax = ax, plot_y = spectrum.y_smooth, color = 'grey')

    
    fit_range_index = [np.where(np.abs(spectrum.x-fit_range[0]) <= 1)[0][0], np.where(np.abs(spectrum.x-fit_range[1]) <= 1)[0][0]]
    
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 3000, fs = 30000)
    peak_fits = spt.approx_peak_gausses(spectrum.x[fit_range_index[0]:fit_range_index[1]], 
                                        spectrum.y_smooth[fit_range_index[0]:fit_range_index[1]], 
                                        smooth_first=False, plot= False, 
                                        height_frac = height_frac, 
                                        threshold=0.1)
    
    for this_peak in peak_fits:
        # this_peak = peak_fits[]
        height = this_peak[0]
        mu = this_peak[1]
        width = this_peak[2]
        sigma = width/2.35
        x = spectrum.x[fit_range_index[0]:fit_range_index[1]]
        y = gaussian(x, height, mu, width, height_frac)
    
        this_peak = Peak(mu = mu, height = height, width = width, sigma = sigma, area = height * sigma * (2*np.pi)**(1/2)) 
        this_peak.spectrum = SERS.SERS_Spectrum(x = x, y = y)    

        this_peak.spectrum.plot(ax = ax)
    
    ax.set_xlim(fit_range)
    plt.show()
    
    
#%% Fit peaks of individual regions - scipy.optimize.curve_fit()

particle = particles[11]
powerseries = particle.powerseries


def gauss(x: np.ndarray, a: float, mu: float, sigma: float, b: float) -> np.ndarray:
    return (
        a/sigma/np.sqrt(2*np.pi)
    )*np.exp(
        -0.5 * ((x-mu)/sigma)**2
    ) + b

# 1620 peak

fit_range = [1351, 1450]

height_frac = 0.5

for i, spectrum in enumerate(powerseries[4:5]):
    
    fit_range_index = [np.where(np.abs(spectrum.x-fit_range[0]) <= 1)[0][0], np.where(np.abs(spectrum.x-fit_range[1]) <= 1)[0][0] + 1]
    
    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)


    X = spectrum.x[fit_range_index[0] : fit_range_index[1]]
    Y = spectrum.y_smooth[fit_range_index[0] : fit_range_index[1]]
    xmin, xmax = X.min(), X.max()  # left and right bounds
    i_max = Y.argmax()             # index of highest value - for guess, assumed to be Gaussian peak
    ymax = Y[i_max]     # height of guessed peak
    mu0 = X[i_max]      # centre x position of guessed peak
    b0 = (Y[0]+Y[len(Y)-1])/2  # height of baseline guess
    i_half = np.argmax(Y >= (ymax + b0)/2)      # Index of first argument to be at least halfway up the estimated bell
    # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
    # half, and even if this point is a distant outlier the fit should still converge.
    sigma0 = (mu0 - X[i_half]) / np.sqrt(2*np.log((ymax - b0)/(Y[i_half] - b0)))
    # sigma0 = 3
    a0 = (ymax - b0) * sigma0 * np.sqrt(2*np.pi)
    p0 = a0, mu0, sigma0, b0
    
    try:
        popt, _ = curve_fit(
            f=gauss, xdata=X, ydata=Y, p0=p0,
            bounds=(
                (     1, xmin,           0,    0),
                (np.inf, xmax, xmax - xmin, ymax),
            ),
        )
    except:
        popt = [a0, mu0, sigma0, b0]
        print(i)
        print('Fit Error')
    # print('Guess:', np.array(p0))
    # print('Fit:  ', popt)
    
    mu = popt[1]

    sigma = popt[2]
    width = sigma * 2.35
    area = popt[0]
    b = popt[3]
    x = X
    y = gauss(X, area, mu, sigma, b)
    
    this_peak = Peak(mu = mu, height = height, width = width, sigma = sigma, area = height * sigma * (2*np.pi)**(1/2)) 
    this_peak.spectrum = SERS.SERS_Spectrum(x = x, y = y)    

    fig, ax = plt.subplots(1,1,figsize=[18,9])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
    ax.set_ylabel('SERS Intensity (cts/mW/s)')
    spectrum.plot(ax = ax, plot_y = spectrum.y_baselined, color = 'black')
    spectrum.plot(ax = ax, plot_y = spectrum.y_smooth)
    this_peak.spectrum.plot(ax = ax)
    ax.set_ylim(Y.min()/1.5, Y.max() * 1.1)
    ax.set_xlim(fit_range)
    
#%% Peak fitting for whole spectrum - unused

# #%% Test spectrum_tools.approx_peak_gausses


# def gaussian(x, height, center, width, width_height_frac = 0.5):
#     a = height
#     b = center
#     c = width/(2*np.sqrt(2*np.log(1/width_height_frac)))
    
#     return a*np.exp(-(((x - b)**2)/(2*c**2)))

# fit_range = [None,None]   
# particle = particles[32]
# powerseries = particle.powerseries
# spectrum = powerseries[9]
# spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 5000, fs = 30000)
# height_frac = 0.01
# start = time.time()
# x = spt.approx_peak_gausses(spectrum.x[fit_range[0]:fit_range[1]], spectrum.y_smooth[fit_range[0]:fit_range[1]], smooth_first=False, plot= False, height_frac = height_frac, threshold=0.1)
# finish = time.time()
# print(finish - start)

# y  = []
# for i, this_peak in enumerate(x):
#     height = x[i][0]
#     mu = x[i][1]
#     width = x[i][2]
#     sigma = width/2.35
#     area =  height * sigma * (2*np.pi)**(1/2) 
#     b = 0
#     # y.append(gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b))
#     y.append(gaussian(wn_cal, height, mu, width, height_frac))

# y = np.array(y)
# y_tot = y.sum(axis = 0)

# residuals = spectrum.y_smooth[fit_range[0]:fit_range[1]] - y_tot[fit_range[0]:fit_range[1]]
# residuals_tot = np.sum(np.abs(residuals))

# fig, ax = plt.subplots(1,1,figsize=[18,9])
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('SERS Intensity (cts/mW/s)')
# fig.suptitle('spt.approx_peak_gausses()')
# ax.plot(wn_cal, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
# for i,peak in enumerate(x):
#     ax.scatter(x[i][1], x[i][0], marker = 'x', s = 100)
#     ax.plot(spectrum.x, y[i], zorder = 1, linestyle = 'dashed')
# ax.plot(spectrum.x, y_tot, color = (1,0,0,0.6), linewidth = 2, label = 'Fit', zorder = 2)
# ax.plot(spectrum.x[fit_range[0]:fit_range[1]], residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
# ax.plot(1,1, label = 'Run time (s): ' + str(finish - start))
# # ax.set_xlim(spectrum.x[fit_range[0]],spectrum.x[fit_range[1]])
# # ax.set_xlim(1350,1500)
# ax.legend()


# #%% Fit peaks for single powerseries - testing quality control


# particle = particles[11]
# powerseries = particle.powerseries

# for j, spectrum in enumerate(powerseries):
    
#     if np.sum(spectrum.y_baselined) <= 1000:
#         print(j)
#         continue
    
#     spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 5000, fs = 30000)
#     # print('\n')
#     # print(spectrum.name)
#     height_frac = 0.2
#     passed = False
#     while passed == False:
#         start = time.time()
#         x = spt.approx_peak_gausses(spectrum.x, spectrum.y_smooth, smooth_first=False, plot= False, height_frac = height_frac, threshold=0.1)
#         finish = time.time()
        
#         y  = []
#         for i, this_peak in enumerate(x):
#             height = x[i][0]
#             mu = x[i][1]
#             width = x[i][2]
#             sigma = width/2.35
#             area =  height * sigma * (2*np.pi)**(1/2) 
#             b = 0
#             # y.append(gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b))
#             y.append(gaussian(spectrum.x, height, mu, width, height_frac))
        
#         y = np.array(y)
#         y_tot = y.sum(axis = 0)
        
#         residuals = spectrum.y_smooth - y_tot
#         residuals_tot = np.sum(np.abs(residuals))
    
#         if np.abs(residuals).max() <= np.percentile(spectrum.y_smooth, 80) and residuals_tot <= np.sum(spectrum.y_smooth)*0.3:
#             passed = True
#             # print('Passed')
#         else:
#             height_frac -= 0.01
            
#         if height_frac <= 0.01:
#             passed = True
#             # print('Failed')
#             continue
    
#     # print(residuals_tot/np.sum(spectrum.y_smooth))
#     # print(np.abs(residuals).max())
#     # print(np.percentile(spectrum.y_smooth, 75))
    
#     fig, ax = plt.subplots(1,1,figsize=[18,9])
#     ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
#     ax.set_ylabel('SERS Intensity (cts/mW/s)')
#     fig.suptitle(spectrum.name)
#     ax.plot(spectrum.x, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
#     for i,peak in enumerate(x):
#         ax.scatter(x[i][1], x[i][0], marker = 'x', s = 100)
#         ax.plot(spectrum.x, y[i], zorder = 1, linestyle = 'dashed')
#     ax.plot(spectrum.x, y_tot, color = (1,0,0,0.6), linewidth = 2, label = 'Fit', zorder = 2)
#     ax.plot(spectrum.x, residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
#     ax.plot(1,1, label = 'Run time (s): ' + str(finish - start))
#     ax.plot(1,1, label = 'Height frac: ' + str(height_frac))
#     # ax.set_xlim(spectrum.x[fit_range[0]],spectrum.x[fit_range[1]])
#     ax.set_xlim(1550,1650)
#     ax.legend()




#%% Fit peaks for all particles

for j, particle in tqdm(enumerate(particles), leave = True):
    
    peaks = []
    height_frac = 0.02
    powerseries = particle.powerseries
    
    for i, spectrum in enumerate(powerseries):
        
        if np.sum(spectrum.y_baselined) <= 1000:
            these_peaks = []
            continue

        spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 5000, fs = 30000)
        
        x = spt.approx_peak_gausses(spectrum.x, spectrum.y_smooth, smooth_first=False, plot= False, height_frac = height_frac, threshold=0.1)

        these_peaks = []
        
        for k in range(0, len(x)-1):
            height = x[k][0]
            mu = x[k][1]
            width = x[k][2]
            sigma = width/2.35
            area =  height * sigma * (2*np.pi)**(1/2) 
            b = 0
            y = gaussian(spectrum.x, height, mu, width, height_frac)
        
            this_peak = Peak(mu, height, width, sigma, area)
            this_peak.y = y
            these_peaks.append(this_peak)

        peaks.append(these_peaks)

    particle.peaks = peaks

#%% Get dark times

dark_times = []

for particle in particles:
    
    if particle.dark_time not in dark_times:
        dark_times.append(particle.dark_time)
    

#%% Get chosen peaks

for particle in particles:

    particle.peak_510 = []
    particle.peak_1280 = []
    particle.peak_1330 = []
    particle.peak_1420 = []    
    particle.peak_1620 = []
    
    for these_peaks in peaks:
        
        for peak in these_peaks:
            
            if peak.mu > 500 and peak.mu < 520 and peak.height > 5000:
                particle.peak_510.append(peak)
            
            if peak.mu > 1280 and peak.mu < 1320 and peak.height > 15000:
                particle.peak_1280.append(peak)
            
            if peak.mu > 1310 and peak.mu < 1350 and peak.height > 10000:
                particle.peak_1330.append(peak)
                
            if peak.mu > 1400 and peak.mu < 1440 and peak.height > 18000:
                particle.peak_1420.append(peak)
            
            if peak.mu > 1600 and peak.height >10000:
                particle.peak_1620.append(peak)

    # if len(particle.peak_510) != 40:
    #     print(particle.name)
    #     print(len(particle.peak_510))

    #     print('510')

    # if len(particle.peak_1280) != 40:
    #     print(particle.name)
    #     print(len(particle.peak_1280))

    #     print('1280')

    # if len(particle.peak_1330) != 40:
    #     print(particle.name)
    #     print(len(particle.peak_1330))
    #     print('1330')                

    if len(particle.peak_1420) != 40:
        print(particle.name)
        print(len(particle.peak_1420))
        print('1420')
        
    if len(particle.peak_1620) != 37:
        print(particle.name)
        print(len(particle.peak_1620))
        print('1620')


#%% Get averages & errors of chosen peaks' amplitude

avg_counter = 0

avg_particle.peak_1280 = np.zeros(40)
avg_particle.peak_1330 = np.zeros(40)
avg_particle.peak_1420 = np.zeros(40)
avg_particle.peak_1620 = np.zeros(40)

avg_1280 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1280 = []
    for j in range(0,40):
        this_avg_1280.append(particle.peak_1280[j].area)
    avg_1280.append(this_avg_1280)
avg_1280 = np.array(avg_1280)
mean_1280 = np.mean(avg_1280, axis = 0)
error_1280 = np.std(avg_1280, axis = 0)/sqrt(31)        

avg_1330 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1330 = []
    for j in range(0,40):
        this_avg_1330.append(particle.peak_1330[j].area)
    avg_1330.append(this_avg_1330)
avg_1330 = np.array(avg_1330)
mean_1330 = np.mean(avg_1330, axis = 0)
error_1330 = np.std(avg_1330, axis = 0)/sqrt(31)    

# avg_1420 = []
# for i, particle in enumerate(particles[0:31]):
#     this_avg_1420 = []
#     for j, spectrum in enumerate(particle.powerseries):
#         this_avg_1420.append(particle.peak_1420[j].area)
#     avg_1420.append(this_avg_1420)
# avg_1420 = np.array(avg_1420)
# mean_1420 = np.mean(avg_1420, axis = 0)
# error_1420 = np.std(avg_1420, axis = 0)/sqrt(31)       

# avg_1620 = []
# for i, particle in enumerate(particles[0:31]):
#     this_avg_1620 = []
#     for j, spectrum in enumerate(particle.powerseries):
#         this_avg_1620.append(particle.peak_1620[j].area)
#     avg_1620.append(this_avg_1620)
# avg_1620 = np.array(avg_1620)
# mean_1620 = np.mean(avg_1620, axis = 0)
# error_1620 = np.std(avg_1620, axis = 0)/sqrt(31)     
            
#%% Plot peak amplitude v. power of MLAgg Avg

# Plot peak amplitude v. power

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Scan No.', size = 'large')
ax.set_ylabel('Peak Intensity', size = 'large')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.set_xticks(scan)
ax.errorbar(scan, mean_1280, yerr = error_1280, marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.errorbar(scan, mean_1330, yerr = error_1330, marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
# ax.errorbar(scan, mean_1420, yerr = error_1420, marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
# ax.errorbar(scan, mean_1620, yerr = error_1620, marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.legend()
fig.suptitle('785nm Powerseries - Peak Amplitude - Full Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Amplitude Full Powerseries' + '.svg', format = 'svg')
# plt.close(fig)

#%%
# Plot peak amplitude v. power - low power only

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Previous Laser Power', size = 'large')
ax.set_ylabel('Peak Intensity', size = 'large')
ax.set_xscale('log')
# ax.set_yscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.errorbar(powers_list[0::2], mean_1280[1::2], yerr = error_1280[1::2], marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.errorbar(powers_list[0::2], mean_1330[1::2], yerr = error_1330[1::2], marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.errorbar(powers_list[0::2], mean_1420[1::2], yerr = error_1420[1::2], marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.errorbar(powers_list[0::2], mean_1620[1::2], yerr = error_1620[1::2], marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.legend(loc = 'upper right')
fig.suptitle('785nm Powerseries - Peak Amplitude - Min Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Amplitude Min Powerseries' + '.svg', format = 'svg')
# plt.close(fig)

#%%
# Plot peak amplitude v. power - direct powerseries

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Laser Power', size = 'large')
ax.set_ylabel('Peak Intensity', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.errorbar(powers_list[0::2], mean_1280[0::2], yerr = error_1280[1::2], marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.errorbar(powers_list[0::2], mean_1330[0::2], yerr = error_1330[1::2], marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.errorbar(powers_list[0::2], mean_1420[0::2], yerr = error_1420[1::2], marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.errorbar(powers_list[0::2], mean_1620[0::2], yerr = error_1620[1::2], marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.legend()
fig.suptitle('785nm Powerseries - Peak Amplitude - Direct Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Amplitude Direct Powerseries' + '.svg', format = 'svg')
# plt.close(fig)


#%% Plot %change peak amplitude v. power of MLAgg Avg

dp1 = (mean_1280[:] - mean_1280[0])/mean_1280[0] * 100
dp2 = (mean_1330[:] - mean_1330[0])/mean_1330[0] * 100
dp3 = (mean_1420[:] - mean_1420[0])/mean_1420[0] * 100
dp4 = (mean_1620[:] - mean_1620[0])/mean_1620[0] * 100

# Plot peak amplitude v. power

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Scan No.', size = 'large')
ax.set_ylabel('$\Delta_{Intensity}$ (%)', size = 'large')
ax.set_xticks(scan[::2])
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(scan, dp1, marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2)
ax.plot(scan, dp2, marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2)  
ax.plot(scan, dp3, marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2)        
ax.plot(scan, dp4, marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2)        

ax.legend()
fig.suptitle('785nm Powerseries - % Change Peak Amplitude - Full Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Change Peak Amplitude Full Powerseries' + '.svg', format = 'svg')
# plt.close(fig)

#%%

# Plot peak amplitude v. power - low power only

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Previous Laser Power', size = 'large')
ax.set_ylabel('$\Delta_{Intensity}$ (%)', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0::2], dp1[1::2], marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp2[1::2], marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp3[1::2], marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp4[1::2], marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2)
ax.legend(loc = 'upper right')
fig.suptitle('785nm Powerseries - % Change Peak Amplitude - Min Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Change Peak Amplitude Min Powerseries' + '.svg', format = 'svg')
# plt.close(fig)

#%%
# Plot peak amplitude v. power - direct powerseries

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Laser Power', size = 'large')
ax.set_ylabel('$\Delta_{Intensity}$ (%)', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0::2], dp1[0::2], marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp2[0::2], marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp3[0::2], marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp4[0::2], marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2)
ax.legend()
fig.suptitle('785nm Powerseries - % Change Peak Amplitude - Direct Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Change Peak Amplitude Direct Powerseries' + '.svg', format = 'svg')
# plt.close(fig)



#%%

#%% Get averages & errors of chosen peaks' position

avg_counter = 0
avg_particle = particles[32]
avg_particle.peak_1280 = np.zeros(len(avg_particle.powerseries))
avg_particle.peak_1330 = np.zeros(len(avg_particle.powerseries))
avg_particle.peak_1420 = np.zeros(len(avg_particle.powerseries))
avg_particle.peak_1620 = np.zeros(len(avg_particle.powerseries))

avg_1280 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1280 = []
    for j, spectrum in enumerate(particle.powerseries):
        this_avg_1280.append(particle.peak_1280[j].mu)
    avg_1280.append(this_avg_1280)
avg_1280 = np.array(avg_1280)
mean_1280 = np.mean(avg_1280, axis = 0)
error_1280 = np.std(avg_1280, axis = 0)/sqrt(31)        

avg_1330 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1330 = []
    for j, spectrum in enumerate(particle.powerseries):
        this_avg_1330.append(particle.peak_1330[j].mu)
    avg_1330.append(this_avg_1330)
avg_1330 = np.array(avg_1330)
mean_1330 = np.mean(avg_1330, axis = 0)
error_1330 = np.std(avg_1330, axis = 0)/sqrt(31)    

avg_1420 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1420 = []
    for j, spectrum in enumerate(particle.powerseries):
        this_avg_1420.append(particle.peak_1420[j].mu)
    avg_1420.append(this_avg_1420)
avg_1420 = np.array(avg_1420)
mean_1420 = np.mean(avg_1420, axis = 0)
error_1420 = np.std(avg_1420, axis = 0)/sqrt(31)       

avg_1620 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1620 = []
    for j, spectrum in enumerate(particle.powerseries):
        this_avg_1620.append(particle.peak_1620[j].mu)
    avg_1620.append(this_avg_1620)
avg_1620 = np.array(avg_1620)
mean_1620 = np.mean(avg_1620, axis = 0)
error_1620 = np.std(avg_1620, axis = 0)/sqrt(31)  
            
#%% Plot peak position v. power


# Plot peak position v. power

dp1 = (mean_1280[:] - mean_1280[0])/mean_1280[0] * 100
dp2 = (mean_1330[:] - mean_1330[0])/mean_1330[0] * 100
dp3 = (mean_1420[:] - mean_1420[0])/mean_1420[0] * 100
dp4 = (mean_1620[:] - mean_1620[0])/mean_1620[0] * 100

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Scan No.', size = 'large')
ax.set_ylabel('Peak Position Change (cm$^{-1}$)', size = 'large')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(scan, dp1, marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2)
ax.plot(scan, dp2, marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2)  
ax.plot(scan, dp3, marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2)        
ax.plot(scan, dp4, marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2)        
ax.legend()
fig.suptitle('785nm Powerseries - Peak Position - Full Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Position Full Powerseries' + '.svg', format = 'svg')
# plt.close(fig)
#%%

# # Plot peak pos v. power - low power only

# particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Previous Laser Power', size = 'large')
ax.set_ylabel('Peak Position Change (cm$^{-1}$)', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0::2], dp1[1::2], marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp2[1::2], marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp3[1::2], marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp4[1::2], marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2)
ax.legend()
fig.suptitle('785nm Powerseries - Peak Position - Min Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Position Min Powerseries' + '.svg', format = 'svg')
# plt.close(fig)
            
#%%
# Plot peak pos v. power - direct powerseries

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Laser Power', size = 'large')
ax.set_ylabel('Peak Position Change (cm$^{-1}$)', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0::2], dp1[0::2], marker = 'o', markersize = 6, color = 'black', linewidth = 1, label = '1280 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp2[0::2], marker = 'o', markersize = 6, color = 'red', linewidth = 1, label = '1330 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp3[0::2], marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp4[0::2], marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2)
ax.legend()
fig.suptitle('785nm Powerseries - Peak Position - Direct Powerseries', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Position Direct Powerseries' + '.svg', format = 'svg')
# plt.close(fig)


#%%

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Scan #', size = 'large')
ax.set_ylabel('Peak Amplitude', size = 'large')
this_particle_list = [particles[0], particles[10], particles[20], particles[30], particles[40], particles[50], particles[60], particles[70], particles[80], particles[90]]
particle = particles[0]
for i in range(0, 5):
    if i % 2 == 0:
        marker = 'x'
    else:
        marker = 'o'
        
    ax.scatter(i, particle.peak_510[i].area, color = 'darkblue', label = '510', marker = marker)
    ax.scatter(i, particle.peak_1280[i].area, color = 'purple', label = '1280', marker = marker)
    ax.scatter(i, particle.peak_1330[i].area, color = 'darkorange', label = '1330', marker = marker)
    ax.scatter(i, particle.peak_1420[i].area, color = 'green', label = '1420', marker = marker)
    ax.scatter(i, particle.peak_1620[i].area, color = 'red', label = '1620', marker = marker)
    
# ax.legend()
fig.suptitle('633nm Powerswitch Recover - Peak Amplitude', fontsize = 'large')
ax.set_title('Co-TAPP-SMe MLAgg - ' + str(np.round(particle.dark_time, 2)) + ' s Recovery Time')


# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Position Direct Powerseries' + '.svg', format = 'svg')
# plt.close(fig)