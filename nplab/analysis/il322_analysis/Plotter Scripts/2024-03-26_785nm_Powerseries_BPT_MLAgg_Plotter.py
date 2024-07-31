# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Co-TAPP-SMe 785nm SERS Powerseries:


Data: 2024-03-26_785nm_Powerseries_BPT_MLAgg.h5


(samples:
     2024-03-18_BPT_60nm_MLAgg_on_Glass)

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



#%% h5 files

## Load raw data h5
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-03-26_785nm_Powerseries_BPT_MLAgg.h5")
dark_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2024-03-26_785_dark_powerseries.h5")


#%% Spectral calibration

# Spectral calibration

## Get default literature BPT spectrum & peaks
lit_spectrum, lit_wn = cal.process_default_lit_spectrum()

## Load BPT ref spectrum
bpt_ref = my_h5['ref_meas']['BPT_785']
bpt_ref = SERS.SERS_Spectrum(bpt_ref)

## Coarse adjustments to miscalibrated spectra
coarse_shift = 70 # coarse shift to ref spectrum
coarse_stretch = 1 # coarse stretch to ref spectrum
notch_range = [(70 + coarse_shift) * coarse_stretch, (128 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
truncate_range = [notch_range[1] + 200, None] # Truncate range for all spectra on this calibration - Add 50 to take out notch slope

## Convert to wn
bpt_ref.x = spt.wl_to_wn(bpt_ref.x, 785)
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
ref_wn = cal.find_ref_peaks(bpt_ref_no_notch, lit_spectrum = lit_spectrum, lit_wn = lit_wn, threshold = 0.2, distance = 1)

ref_wn[3] = bpt_ref_no_notch.x[373]

## Find calibrated wavenumbers
wn_cal = cal.calibrate_spectrum(bpt_ref_no_notch, ref_wn, lit_spectrum = lit_spectrum, lit_wn = lit_wn, linewidth = 1, deg = 2)
bpt_ref.x = wn_cal

## Save to h5
# try:
#     save_h5.create_group('calibration')
# except:
#     pass
# save_group = save_h5['calibration']

# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('SERS Intensity (cts/mW/s)')
# ax.plot(bpt_ref.x, bpt_ref.y, color = 'pink')

# # img = fig2img(fig)


# # dset = save_h5.create_dataset('BPT_ref', data = img)

# # dset.attrs["DISPLAY_ORIGIN"] = np.string_("UL")



#%% Spectral efficiency white light calibration

white_ref = my_h5['ref_meas']['white_scatt_785']
white_ref = SERS.SERS_Spectrum(white_ref.attrs['wavelengths'], white_ref[2], title = 'White Scatterer')

## Convert to wn
white_ref.x = spt.wl_to_wn(white_ref.x, 785)
white_ref.x = white_ref.x + coarse_shift
white_ref.x = white_ref.x * coarse_stretch

## Get white bkg (counts in notch region)
#notch = SERS.SERS_Spectrum(white_ref.x[np.where(white_ref.x < (notch_range[1]-50))], white_ref.y[np.where(white_ref.x < (notch_range[1] - 50))], name = 'White Scatterer Notch') 
# notch = SERS.SERS_Spectrum(x = spt.truncate_spectrum(white_ref.x, white_ref.y, notch_range[0], notch_range[1] - 100)[0], 
#                             y = spt.truncate_spectrum(white_ref.x, white_ref.y, notch_range[0], notch_range[1] - 100)[1], 
#                             name = 'White Scatterer Notch')
notch = SERS.SERS_Spectrum(white_ref.x, white_ref.y, title = 'Notch')
notch_range = [(70 + coarse_shift) * coarse_stretch, (115 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
notch.truncate(notch_range[0], notch_range[1])
notch_cts = notch.y.mean()
notch.plot(title = 'White Scatter Notch')


# ## Truncate out notch (same as BPT ref), assign wn_cal
white_ref.truncate(start_x = truncate_range[0], end_x = truncate_range[1])


## Convert back to wl for efficiency calibration
white_ref.x = spt.wn_to_wl(white_ref.x, 785)


# Calculate R_setup

R_setup = cal.white_scatter_calibration(wl = white_ref.x,
                                    white_scatter = white_ref.y,
                                    white_bkg = -10000,
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
        
    directory_path = r'C:\Users\il322\Desktop\Offline Data\2024-03-26 Analysis\_' + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")

    return directory_path


#%% 785nm MLAGG dark counts

particle = dark_h5['PT_lab']


# Add all SERS spectra to powerseries list in order

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []
for key in keys:
    if 'dark_powerseries' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.x = spt.wl_to_wn(spectrum.x, 785)
    spectrum.x = spectrum.x + coarse_shift
    spectrum.x = spectrum.x * coarse_stretch
    spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    spectrum.x = wn_cal
    spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 0.1, cutoff = 1500, fs = 60000)
    powerseries[i] = spectrum
    
dark_powerseries = powerseries

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')

for spectrum in dark_powerseries:
    ax.plot(spectrum.x, spectrum.y)

dark_powerseries = np.insert(dark_powerseries,np.arange(1,11,1), dark_powerseries[0])

dark_powerseries = dark_powerseries[0:16]

# List of powers used, for colormaps

powers_list = []
# colors_list = np.arange(0,15,1)

for spectrum in dark_powerseries:
    powers_list.append(spectrum.laser_power)
    print(spectrum.cycle_time)
    
powers_list[14] = 3.2
    

#%% Testing background subtraction


particle = my_h5['ParticleScannerScan_1']['Particle_0']


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


for i, spectrum in enumerate(powerseries[0:10]):
    

    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.x = spt.wl_to_wn(spectrum.x, 785)
    spectrum.x = spectrum.x + coarse_shift
    spectrum.x = spectrum.x * coarse_stretch
    spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    spectrum.x = wn_cal
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_powerseries[i].y,
                                  exposure = spectrum.cycle_time)
    
    spectrum.y = spt.remove_cosmic_rays(spectrum.y)
    # spectrum.truncate(450, truncat)
    
    ## Baseline
    spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    spectrum.y_baselined = spectrum.y - spectrum.baseline
    
    ## Plot raw, baseline, baseline subtracted
    spectrum.plot(ax = ax, plot_y = (spectrum.y - spectrum.y.min()), linewidth = 1, color = 'black', label = i, zorder = 30-i)
    spectrum.plot(ax = ax, plot_y = spectrum.y_baselined, linewidth = 1, color = 'purple', label = i, zorder = 30-i)
    spectrum.plot(ax = ax, plot_y = spectrum.baseline - spectrum.y.min(), color = 'darkred', linewidth = 1)    
    
    ## Labeling & plotting
    # ax.legend(fontsize = 18, ncol = 5, loc = 'upper center')
    # ax.get_legend().set_title('Scan No.')
    # for line in ax.get_legend().get_lines():
    #     line.set_linewidth(4.0)
    fig.suptitle('MLAgg')
    powerseries[i] = spectrum
    
    # ax.set_xlim(1200, 1700)
    # ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
    plt.tight_layout(pad = 0.8)
    
    
#%% Plot baseline-subtracted min powerseries & direct powerseries for each MLAGG spot

scan_list = ['ParticleScannerScan_1']

avg_powerseries = np.zeros([len(dark_powerseries), len(spectrum.y)])
avg_counter = 0

# Loop over particles in particle scan

for particle_scan in scan_list:
    particle_list = []
    particle_list = natsort.natsorted(list(my_h5[particle_scan].keys()))
    
    ## Loop over particles in particle scan
    for particle in particle_list:
        if 'Particle' not in particle:
            particle_list.remove(particle)

    
    # Loop over particles in particle scan
    
    for particle in particle_list:
        avg_counter += 1
        particle_name = 'MLAgg_' + str(particle_scan) + '_' + particle
        particle = my_h5[particle_scan][particle]
        print('\n' + particle_name)
    
        # Add all SERS spectra to powerseries list in order
        
        keys = list(particle.keys())
        keys = natsort.natsorted(keys)
        powerseries = []
        
        for key in keys:
            if 'SERS' in key:
                powerseries.append(particle[key])
        
        powerseries = powerseries[0:16]
        
        fig, ax = plt.subplots(1,1,figsize=[12,9])
        ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
        ax.set_ylabel('SERS Intensity (cts/mW/s)')
        
        fig2, ax2 = plt.subplots(1,1,figsize=[12,9])
        ax2.set_xlabel('Raman Shifts (cm$^{-1}$)')
        ax2.set_ylabel('SERS Intensity (cts/mW/s)')
        
        powerseries_y = np.zeros((len(powerseries), len(spectrum.y)))
        
        for i, spectrum in enumerate(powerseries):
            
            ## x-axis truncation, calibration
            spectrum = SERS.SERS_Spectrum(spectrum)
            spectrum.x = spt.wl_to_wn(spectrum.x, 785)
            spectrum.x = spectrum.x + coarse_shift
            spectrum.x = spectrum.x * coarse_stretch
            spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
            spectrum.x = wn_cal
            if i == 14:
                spectrum.laser_power = 3.2
                print(spectrum.laser_power)
            spectrum.calibrate_intensity(R_setup = R_setup,
                                          dark_counts = dark_powerseries[i].y,
                                          exposure = spectrum.cycle_time,
                                          laser_power = spectrum.laser_power)
            
            # spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 10)
            # spectrum.truncate(450, 1500)
            
            ## Baseline
            spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
            spectrum.y_baselined = spectrum.y - spectrum.baseline
            
            # spectrum.normalise(norm_y = spectrum.y)
            avg_powerseries[i] += spectrum.y_baselined
            
            # Plot min powerseries
            
            my_cmap = plt.get_cmap('inferno')
            color = my_cmap(i/16)
            offset = 0
            if i == 0:
                previous_power = '0.0'
            else:
                previous_power = np.round(powerseries[i-1].laser_power * 1000, 0)
            if spectrum.laser_power <= 0.0029:
                spectrum.plot(ax = ax, plot_y = spectrum.y_baselined + (i*offset), title = '785nm Min Power Powerseries - Co-TAPP-SMe 60nm MLAgg', linewidth = 1, color = color, label = previous_power, zorder = (19-i))

            ## Labeling & plotting
            ax.legend(fontsize = 18, ncol = 5, loc = 'upper center')
            ax.get_legend().set_title('Previous laser power ($\mu$W)')
            for line in ax.get_legend().get_lines():
                line.set_linewidth(4.0)
            fig.suptitle(particle.name)
            powerseries[i] = spectrum
            
            # ax.set_xlim(550, 1500)
            # ax.set_ylim(-500, 13000)
            plt.tight_layout(pad = 0.8)
        
            
            # Plot direct powerseries
            
            my_cmap = plt.get_cmap('inferno')
            color = my_cmap(i/16)
            offset = 0
            if i%2 == 0:
                spectrum.plot(ax = ax2, plot_y = spectrum.y_baselined + (i*offset), title = '785nm Direct Power Powerseries - Co-TAPP-SMe 60nm MLAgg', linewidth = 1, color = color, label = np.round(spectrum.laser_power * 1000, 0), zorder = (19-i))
        
            ## Labeling & plotting
            ax2.legend(fontsize = 18, ncol = 5, loc = 'upper center')
            ax2.get_legend().set_title('Laser power ($\mu$W)')
            for line in ax2.get_legend().get_lines():
                line.set_linewidth(4.0)
            fig2.suptitle(particle.name)
            powerseries[i] = spectrum
            
            # ax2.set_xlim(550, 1500)
            # ax2.set_ylim(-500, 13000)
            plt.tight_layout(pad = 0.8)
            
            
            # Plot timescan powerseries
            
            powerseries_y[i] = spectrum.y_baselined
        powerseries_y = np.array(powerseries_y)
        timescan = SERS.SERS_Timescan(x = spectrum.x, y = powerseries_y, exposure = 1)
        fig3, (ax3) = plt.subplots(1, 1, figsize=[12,16])
        t_plot = np.arange(0,len(powerseries),1)
        v_min = powerseries_y.min()
        v_max = np.percentile(powerseries_y, 99.9)
        cmap = plt.get_cmap('inferno')
        ax3.set_yticklabels([])
        ax3.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
        # ax3.set_xlim(450,1500)
        ax3.set_title('785nm Powerseries' + 's\n' + str(particle_name), fontsize = 'x-large', pad = 10)
        pcm = ax3.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
        clb = fig3.colorbar(pcm, ax=ax3)
        clb.set_label(label = 'SERS Intensity', size = 'large', rotation = 270, labelpad=30)
            
            
        # # Save plots
        
        # save_dir = get_directory(particle_name)
        # fig.savefig(save_dir + particle_name + '785nm Min Powerseries' + '.svg', format = 'svg')
        # plt.close(fig)
        # print('Min Powerseries')
        
        # save_dir = get_directory(particle_name)
        # fig2.savefig(save_dir + particle_name + '785nm Direct Powerseries' + '.svg', format = 'svg')
        # plt.close(fig2)
        # print('Direct Powerseries')
        
        # save_dir = get_directory(particle_name)
        # fig3.savefig(save_dir + particle_name + '785nm Powerseries Timescan' + '.svg', format = 'svg')
        # plt.close(fig)
        # print('Powerseries Timescan')
        
            
## Calculate average
for i in range(0, len(avg_powerseries)):
    avg_powerseries[i] = avg_powerseries[i]/avg_counter 
wn_cal_trunc = spectrum.x
    
#%% Plot avg powerseries

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')

fig2, ax2 = plt.subplots(1,1,figsize=[12,9])
ax2.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax2.set_ylabel('SERS Intensity (cts/mW/s)')

for i, spectrum in enumerate(avg_powerseries):
      
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(x = wn_cal_trunc, y = spectrum)
    spectrum.y_baselined = spectrum.y
    
    # Plot min powerseries
    
    my_cmap = plt.get_cmap('inferno')
    color = my_cmap(i/16)
    offset = 0
    if i == 0:
        previous_power = '0.0'
    else:
        previous_power = np.round(powers_list[i-1] * 1000, 0)
    if powers_list[i] <= 0.0029:
        spectrum.plot(ax = ax, plot_y = spectrum.y_baselined + (i*offset), title = '785nm Min Power Powerseries - Co-TAPP-SMe 60nm MLAgg', linewidth = 1, color = color, label = previous_power, zorder = (19-i))

    ## Labeling & plotting
    ax.legend(fontsize = 18, ncol = 5, loc = 'upper center')
    ax.get_legend().set_title('Previous laser power ($\mu$W)')
    for line in ax.get_legend().get_lines():
        line.set_linewidth(4.0)
    fig.suptitle(particle_name)
    # powerseries[i] = spectrum
    
#     ax.set_xlim(550, 1500)
#     ax.set_ylim(-500, 13000)
    plt.tight_layout(pad = 0.8)

    
    # Plot direct powerseries
    
    my_cmap = plt.get_cmap('inferno')
    color = my_cmap(i/16)
    offset = 0
    if i%2 == 0:
        spectrum.plot(ax = ax2, plot_y = spectrum.y_baselined + (i*offset), title = '785nm Direct Power Powerseries - Co-TAPP-SMe 60nm MLAgg', linewidth = 1, color = color, label = np.round(powers_list[i] * 1000, 0), zorder = (19-i))

    ## Labeling & plotting
    ax2.legend(fontsize = 18, ncol = 5, loc = 'upper center')
    ax2.get_legend().set_title('Laser power ($\mu$W)')
    for line in ax.get_legend().get_lines():
        line.set_linewidth(4.0)
    fig2.suptitle(particle_name)    
    # ax2.set_xlim(550, 1500)
    # ax2.set_ylim(-500, 13000)
    plt.tight_layout(pad = 0.8)
    
    # Plot timescan powerseries
    
    powerseries_y[i] = spectrum.y_baselined
powerseries_y = np.array(powerseries_y)
timescan = SERS.SERS_Timescan(x = spectrum.x, y = powerseries_y, exposure = 1)
fig3, (ax3) = plt.subplots(1, 1, figsize=[12,16])
t_plot = np.arange(0,len(powerseries),1)
v_min = powerseries_y.min()
v_max = np.percentile(powerseries_y, 99.9)
cmap = plt.get_cmap('inferno')
ax3.set_yticklabels([])
ax3.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
# ax3.set_xlim(450,1500)
ax3.set_title('785nm Powerseries' + 's\n' + str(particle_name), fontsize = 'x-large', pad = 10)
pcm = ax3.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
clb = fig3.colorbar(pcm, ax=ax3)
clb.set_label(label = 'SERS Intensity', size = 'large', rotation = 270, labelpad=30)
    
# # Save plots

# save_dir = get_directory(particle_name)
# fig.savefig(save_dir + particle_name + '785nm Min Powerseries' + '.svg', format = 'svg')
# plt.close(fig)
# print('Min Powerseries')

# save_dir = get_directory(particle_name)
# fig2.savefig(save_dir + particle_name + '785nm Direct Powerseries' + '.svg', format = 'svg')
# plt.close(fig2)
# print('Direct Powerseries')

# save_dir = get_directory(particle_name)
# fig3.savefig(save_dir + particle_name + '785nm Powerseries Timescan' + '.svg', format = 'svg')
# plt.close(fig)
# print('Powerseries Timescan')

#%% Peak fitting for MLAgg Avg

def gauss(x: np.ndarray, a: float, mu: float, sigma: float, b: float) -> np.ndarray:
    return (
        a/sigma/np.sqrt(2*np.pi)
    )*np.exp(
        -0.5 * ((x-mu)/sigma)**2
    ) + b

        
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# particle = my_h5['ParticleScannerScan_2']['Particle_0']


# # Add all SERS spectra to powerseries list in order

# keys = list(particle.keys())
# keys = natsort.natsorted(keys)
# powerseries = []

# for key in keys:
#     if 'SERS' in key:
#         powerseries.append(particle[key])




# for i, spectrum in enumerate(powerseries):

#     ## x-axis truncation, calibration
#     spectrum = SERS.SERS_Spectrum(spectrum)
#     spectrum.x = spt.wl_to_wn(spectrum.x, 785)
#     spectrum.x = spectrum.x + coarse_shift
#     spectrum.x = spectrum.x * coarse_stretch
#     spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
#     spectrum.x = wn_cal
#     spectrum.calibrate_intensity(R_setup = R_setup,
#                                   dark_counts = dark_powerseries[i].y,
#                                   exposure = spectrum.cycle_time,
#                                   laser_power = spectrum.laser_power)
    
#     spectrum.y = spt.remove_cosmic_rays(spectrum.y)
#     # spectrum.truncate(450, 1500)
#     spectrum.truncate(1090, 1140)
    
def single_gauss(start, stop):
    
    fit = []
    
    fig, ax = plt.subplots(1,1,figsize=[12,9])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
    ax.set_ylabel('SERS Intensity (cts/mW/s)')

    for i, spectrum in enumerate(avg_powerseries):
        
        ## x-axis truncation, calibration
        spectrum = SERS.SERS_Spectrum(x = wn_cal_trunc, y = spectrum)
        spectrum.y_baselined = spectrum.y
        
        start_i = find_nearest(spectrum.x, start)
        stop_i = find_nearest(spectrum.x, stop)
    
        X = spectrum.x[start_i:stop_i]
        Y = spectrum.y_baselined[start_i:stop_i]
        xmin, xmax = X.min(), X.max()  # left and right bounds
        i_max = Y.argmax()             # index of highest value - for guess, assumed to be Gaussian peak
        ymax = Y[i_max]     # height of guessed peak
        mu0 = X[i_max]      # centre x position of guessed peak
        b0 = Y[:20].mean()  # height of baseline guess
        
        # https://en.wikipedia.org/wiki/Gaussian_function#Properties
        # Index of first argument to be at least halfway up the estimated bell
        i_half = np.argmax(Y >= (ymax + b0)/2)
        # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
        # half, and even if this point is a distant outlier the fit should still converge.
        sigma0 = (mu0 - X[i_half]) / np.sqrt(
            2*np.log(
                (ymax - b0)/(Y[i_half] - b0)
            )
        )
        
        a0 = (ymax - b0) * sigma0 * np.sqrt(2*np.pi)
        p0 = a0, mu0, sigma0, b0
        
        popt, _ = curve_fit(
            f=gauss, xdata=X, ydata=Y, p0=p0,
            bounds=(
                (     1, xmin,           0,    0),
                (np.inf, xmax, xmax - xmin, ymax),
            ),
        )
        print('Guess:', np.array(p0))
        print('Fit:  ', popt)
        
        # fig, ax = plt.subplots()
        # ax.set_title('Gaussian fit')
        # ax.scatter(X, Y, marker='+', label='experiment', color='orange')
        # ax.plot(X, gauss(X, *p0), label='guess', color='lightgrey')
        # ax.plot(X, gauss(X, *popt), label='fit')
        # ax.legend()
        # plt.show()
        # if powers_list[i] > 0.0029:
        my_cmap = plt.get_cmap('inferno')
        color = my_cmap(i/16)
        ax.plot(X, Y, color = color, linewidth = 1)
        ax.plot(X, gauss(X, *popt), color = color, linewidth = 1, linestyle = 'dashed', label = powers_list[i])
        # ax.legend()
    
        fit.append(popt)
        
    fit = np.array(fit)
    return fit
    

peak_1 = single_gauss(start = 1150, stop = 1200)
peak_2 = single_gauss(start = 1330, stop = 1365)
peak_3 = single_gauss(start = 1580, stop = 1600)


#%% Plot peak amplitude v. power of MLAgg Avg

# Plot peak amplitude v. power

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Scan No.', size = 'large')
ax.set_ylabel('Peak Intensity', size = 'large')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(scan, peak_1[:,0], marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(scan, peak_2[:,0], marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)  
ax.plot(scan, peak_3[:,0], marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)        

ax.legend()
fig.suptitle('785nm Powerseries - Peak Amplitude - Full Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Amplitude Full Powerseries' + '.svg', format = 'svg')
# plt.close(fig)


# # Plot peak amplitude v. power - low power only

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Previous Laser Power', size = 'large')
ax.set_ylabel('Peak Intensity', size = 'large')
ax.set_xscale('log')
# ax.set_yscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0:14:2], peak_1[1:15:2,0], marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0:14:2], peak_2[1:15:2,0], marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0:14:2], peak_3[1:15:2,0], marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)
ax.legend(loc = 'upper left')
fig.suptitle('785nm Powerseries - Peak Amplitude - Min Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')
ax.set_ylim(15000,90000)

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Amplitude Min Powerseries' + '.svg', format = 'svg')
# plt.close(fig)


# # Plot peak amplitude v. power - direct powerseries

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Laser Power', size = 'large')
ax.set_ylabel('Peak Intensity', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0::2], peak_1[0::2,0], marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], peak_2[0::2,0], marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], peak_3[0::2,0], marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)
ax.legend()
fig.suptitle('785nm Powerseries - Peak Amplitude - Direct Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Amplitude Direct Powerseries' + '.svg', format = 'svg')
# plt.close(fig)


#%% Plot %change peak amplitude v. power of MLAgg Avg

dp1 = (peak_1[:,0] - peak_1[0,0])/peak_1[0,0] * 100
dp2 = (peak_2[:,0] - peak_2[0,0])/peak_2[0,0] * 100
dp3 = (peak_3[:,0] - peak_3[0,0])/peak_3[0,0] * 100

# Plot peak amplitude v. power

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Scan No.', size = 'large')
ax.set_ylabel('$\Delta_{Intensity}$ (%)', size = 'large')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(scan, dp1, marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(scan, dp2, marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)  
ax.plot(scan, dp3, marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)        

ax.legend()
fig.suptitle('785nm Powerseries - % Change Peak Amplitude - Full Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Change Peak Amplitude Full Powerseries' + '.svg', format = 'svg')
# plt.close(fig)


# Plot peak amplitude v. power - low power only

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Previous Laser Power', size = 'large')
ax.set_ylabel('$\Delta_{Intensity}$ (%)', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0:14:2], dp1[1:15:2], marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0:14:2], dp2[1:15:2], marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0:14:2], dp3[1:15:2], marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)
ax.legend(loc = 'upper left')
fig.suptitle('785nm Powerseries - % Change Peak Amplitude - Min Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Change Peak Amplitude Min Powerseries' + '.svg', format = 'svg')
# plt.close(fig)


# Plot peak amplitude v. power - direct powerseries

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Laser Power', size = 'large')
ax.set_ylabel('$\Delta_{Intensity}$ (%)', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0::2], dp1[0::2], marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp2[0::2], marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp3[0::2], marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)
ax.legend()
fig.suptitle('785nm Powerseries - % Change Peak Amplitude - Direct Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Change Peak Amplitude Direct Powerseries' + '.svg', format = 'svg')
# plt.close(fig)
            
#%% Plot peak position v. power


# Plot peak position v. power

dp1 = peak_1[:,1] - peak_1[0,1]
dp2 = peak_2[:,1] - peak_2[0,1]
dp3 = peak_3[:,1] - peak_3[0,1]

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Scan No.', size = 'large')
ax.set_ylabel('Peak Position Change (cm$^{-1}$)', size = 'large')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(scan, dp1, marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(scan, dp2, marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)  
ax.plot(scan, dp3, marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)        
ax.legend()
fig.suptitle('785nm Powerseries - Peak Position - Full Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Position Full Powerseries' + '.svg', format = 'svg')
# plt.close(fig)


# # Plot peak pos v. power - low power only

# particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Previous Laser Power', size = 'large')
ax.set_ylabel('Peak Position Change (cm$^{-1}$)', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0::2], dp1[1::2], marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp2[1::2], marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp3[1::2], marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)
ax.legend()
fig.suptitle('785nm Powerseries - Peak Position - Min Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Position Min Powerseries' + '.svg', format = 'svg')
# plt.close(fig)
            

# Plot peak pos v. power - direct powerseries

particle_name = 'MLAgg_Avg'

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Laser Power', size = 'large')
ax.set_ylabel('Peak Position Change (cm$^{-1}$)', size = 'large')
ax.set_xscale('log')
scan = np.arange(0,len(dark_powerseries),1, dtype = int)   
ax.plot(powers_list[0::2], dp1[0::2], marker = 'o', markersize = 6, color = 'green', linewidth = 1, label = '1175 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp2[0::2], marker = 'o', markersize = 6, color = 'darkorange', linewidth = 1, label = '1350 cm$^{-1}$', zorder = 2)
ax.plot(powers_list[0::2], dp3[0::2], marker = 'o', markersize = 6, color = 'darkblue', linewidth = 1, label = '1600 cm$^{-1}$', zorder = 2)
ax.legend()
fig.suptitle('785nm Powerseries - Peak Position - Direct Powerseries', fontsize = 'large')
ax.set_title('BPT MLAgg Avg')

# ## Save plot
# save_dir = get_directory(particle_name)
# plt.savefig(save_dir + particle_name + 'Peak Position Direct Powerseries' + '.svg', format = 'svg')
# plt.close(fig)
