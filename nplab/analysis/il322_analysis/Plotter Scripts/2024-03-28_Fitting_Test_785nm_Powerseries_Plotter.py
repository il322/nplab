# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Co-TAPP-SMe 785nm SERS Powerseries, using it to experiment with different SERS fitting methods


Data: 2024-03-28_785nm-Powerseries-KineticScan_NIR-Objective_Test.h5


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
        self.peaks = np.zeros((20,5)) 
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
my_h5 = h5py.File(r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-03-28_785nm_Powerseries_KineticScan.h5")



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

ref_wn[3] = bpt_ref_no_notch.x[371]

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

white_ref = my_h5['ref_meas']['white_scatt_x5']
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
notch_range = [(70 + coarse_shift) * coarse_stretch, (105 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
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
        
    directory_path = r'C:\Users\il322\Desktop\Offline Data\2024-03-28 Analysis\_' + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path



#%% 785nm MLAGG dark counts

dark_h5 = h5py.File(r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-03-26_785_dark_powerseries.h5")
particle = dark_h5['PT_lab']


# Add all SERS spectra to powerseries list in order

keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []
dark_powerseries = []
for key in keys:
    if 'new_dark_powerseries' in key:
        powerseries.append(particle[key])
        
for i, spectrum in enumerate(powerseries):
    
    ## x-axis truncation, calibration
    spectrum = SERS.SERS_Spectrum(spectrum)
    spectrum.x = spt.wl_to_wn(spectrum.x, 785)
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

dark_powerseries = np.insert(dark_powerseries, np.arange(1,len(dark_powerseries)+1,1), dark_powerseries[0])

## List of powers used, for colormaps
powers_list = []
for spectrum in dark_powerseries:
    powers_list.append(spectrum.laser_power)
    print(spectrum.cycle_time)
    

#%% Testing background subtraction


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


for i, spectrum in enumerate(powerseries[0:len(powerseries):2]):
  
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
    # spectrum.truncate(450, 1500)

    ## Baseline
    spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    spectrum.y_baselined = spectrum.y - spectrum.baseline
    
    ## Plot raw, baseline, baseline subtracted
    offset = 0
    spectrum.plot(ax = ax, plot_y = (spectrum.y - spectrum.y.min()) + (i*offset), linewidth = 1, color = 'black', label = i, zorder = 30-i)
    spectrum.plot(ax = ax, plot_y = spectrum.y_baselined + (i*offset), linewidth = 1, color = 'purple', label = i, zorder = 30-i)
    spectrum.plot(ax = ax, plot_y = spectrum.baseline- spectrum.y.min() + (i*offset), title = 'Background Subtraction Test', color = 'darkred', linewidth = 1)    
    
    ## Labeling & plotting
    # ax.legend(fontsize = 18, ncol = 5, loc = 'upper center')
    # ax.get_legend().set_title('Scan No.')
    # for line in ax.get_legend().get_lines():
    #     line.set_linewidth(4.0)
    fig.suptitle(particle.name)
    powerseries[i] = spectrum
    
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
    
    for particle in particle_list:
        
        ## Save to class and add to list
        this_particle = Particle()
        this_particle.name = 'MLAgg_' + str(particle_scan) + '_' + particle
        this_particle.h5_address = my_h5[particle_scan][particle]
        particles.append(this_particle)


# Make avg particle

avg_particle = Particle()
avg_particle.h5_address = particles[0].h5_address
avg_particle.name = 'MLAgg_Avg'    
particles.append(avg_particle)

#%% Add & process SERS powerseries for each particle


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
        spectrum.x = spt.wl_to_wn(spectrum.x, 785)
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
    

# Loop over all particles

for particle in tqdm(particles, leave = True):
    
    powerseries = process_powerseries(particle)
    particle.powerseries = powerseries


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
        fig3.savefig(save_dir + particle.name + '785nm Powerseries Timescan' + '.svg', format = 'svg')
        plt.close(fig)


# Loop over all particles and plot

for particle in tqdm(particles, leave = True):
    
    ## Get all specrta into single array for timescan
    powerseries_y = np.zeros((len(particle.powerseries), len(particle.powerseries[0].y)))
    for i,spectrum in enumerate(particle.powerseries):
        powerseries_y[i] = spectrum.y_baselined
    particle.powerseries_y = np.array(powerseries_y)
    
    plot_min_powerseries(particle)
    plot_direct_powerseries(particle)
    plot_timescan_powerseries(particle)


#%% Peak fitting functions


def gauss(x: np.ndarray, a: float, mu: float, sigma: float, b: float) -> np.ndarray:
    return (
        a/sigma/np.sqrt(2*np.pi)
    )*np.exp(
        -0.5 * ((x-mu)/sigma)**2
    ) + b
        

def lorentzian(x, height, center, fwhm):
    I = height
    x0 = center
    gamma = fwhm/2
    numerator = gamma**2
    denominator = (x - x0)**2 + gamma**2
    quot = numerator/denominator
    
    y = I*quot
    return y


#%% Testing scipy.find_peaks()
        

# Find all peaks with scipy.find_peaks()

particle = particles[32]
powerseries = particle.powerseries
spectrum = powerseries[0]
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)
peak_fits, properties = scipy.signal.find_peaks(spectrum.y_smooth,
                                height=spectrum.y_smooth.max()/30,
                                threshold = None,
                                distance = 1,
                                prominence = 0.01,
                                width = 0.01,
                                wlen = None,
                                rel_height = 0.5,
                                plateau_size = None)

peaks = []
for i, peak in enumerate(peak_fits):
    mu = wn_cal[peak-1]
    width = properties['widths'][i]
    height = properties['peak_heights'][i]
    sigma = width/2.35
    area =  height * sigma * (2*np.pi)**(1/2)   
    b = (spectrum.y_smooth[properties['right_bases'][i]] + spectrum.y_smooth[properties['left_bases'][i]])/2
    b = 0
    this_peak = Peak(mu = mu,
                     height = height,
                     width = width,
                     sigma = sigma, 
                     area = area)
    this_peak.x = wn_cal
    this_peak.y = gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b)
    
    peaks.append(this_peak)

y = []
for peak in peaks:
    y.append(peak.y)
y = np.array(y)
y_tot = np.sum(y, axis = 0)
residuals = spectrum.y_smooth - y_tot
residuals_tot = np.sum(np.abs(residuals))
   
# Plot

fig, ax = plt.subplots(1,1,figsize=[18,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
fig.suptitle('Scipy.find_peaks()')
ax.plot(wn_cal, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
for i,peak in enumerate(peaks):
    ax.plot(wn_cal[properties['left_bases'][i]:properties['right_bases'][i]], peak.y[properties['left_bases'][i]:properties['right_bases'][i]], linewidth = 2, linestyle = 'dashed')
    ax.scatter(peak.mu, peak.height, s = 100)
    # ax.scatter(old_peaks[i].mu, old_peaks[i].height)
ax.scatter(spectrum.x[360], spectrum.y_smooth[360], marker= 'x', s = 400)
ax.plot(spectrum.x, y_tot, color = (1,0,0,0.5), linewidth = 2, label = 'Fit')
ax.plot(spectrum.x, residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
ax.set_xlim(1200,1500)
ax.legend()


#%% Testing scipy.find_peaks() + curve_fit()
        

# Find all peaks with scipy.find_peaks()

particle = particles[32]
powerseries = particle.powerseries
spectrum = powerseries[0]
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)
peak_fits, properties = scipy.signal.find_peaks(spectrum.y_smooth,
                                height=spectrum.y_smooth.max()/30,
                                threshold = None,
                                distance = 1,
                                prominence = 0.01,
                                width = 0.01,
                                wlen = None,
                                rel_height = 0.5,
                                plateau_size = None)

peaks = []
for i, peak in enumerate(peak_fits):
    mu = wn_cal[peak-1]
    width = properties['widths'][i]
    height = properties['peak_heights'][i]
    sigma = width/2.35
    area =  height * sigma * (2*np.pi)**(1/2)   
    # b = (spectrum.y_smooth[properties['right_bases'][i]] + spectrum.y_smooth[properties['left_bases'][i]])/2
    b = 0
    this_peak = Peak(mu = mu,
                     height = height,
                     width = width,
                     sigma = sigma, 
                     area = area)
    this_peak.x = wn_cal
    this_peak.y = gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b)
    
    peaks.append(this_peak)


# Curve fit each found peak, using scipy.find_peaks() results as initial guess

for i, peak in enumerate(peaks):
    
    p0 = peak.area, peak.mu, peak.sigma, 0
    xmin = wn_cal[properties['left_bases'][i]]
    xmax = wn_cal[properties['right_bases'][i]]
    try:
        popt, nums = curve_fit(f=gauss, 
                                xdata=wn_cal, 
                                ydata=spectrum.y_smooth, 
                                p0=p0,
                                bounds=((1, xmin, 0, 0),
                                (peak.area * 1.5, xmax, peak.sigma * 1.5, 500),
                                ),)
        
        peak.area = popt[0]
        peak.mu = popt[1]
        peak.sigma = popt[2]
        peak.b = popt[3]
        peak.height = peak.area/(peak.sigma * (2*np.pi)**(1/2))
        peak.width = peak.sigma * 2.35
        peak.y = gauss(wn_cal, *popt)
        
    except:
        print(i)
        print('xo is infeasible')    
        print('\n')
   
y = []
for peak in peaks:
    y.append(peak.y)
y = np.array(y)
y_tot = np.sum(y, axis = 0)
# y_tot = y_tot - 3500
residuals = spectrum.y_smooth - y_tot
residuals_tot = np.sum(np.abs(residuals))
   
# Plot

fig, ax = plt.subplots(1,1,figsize=[18,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
fig.suptitle('curve_fit()')
ax.plot(wn_cal, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
for i,peak in enumerate(peaks):
    ax.plot(wn_cal[properties['left_bases'][i]:properties['right_bases'][i]], peak.y[properties['left_bases'][i]:properties['right_bases'][i]], linewidth = 2, linestyle = 'dashed')
    ax.scatter(peak.mu, peak.height, s = 100)
    # ax.scatter(old_peaks[i].mu, old_peaks[i].height)
ax.scatter(spectrum.x[360], spectrum.y_smooth[360], marker= 'x', s = 400)
ax.plot(spectrum.x, y_tot, color = (1,0,0,0.5), linewidth = 2, label = 'Fit')
ax.plot(spectrum.x, residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
ax.set_xlim(1200,1500)
ax.legend()


#%% Test afr

fit_range = [450,550]   
particle = particles[32]
powerseries = particle.powerseries
spectrum = powerseries[0]
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)

import nplab.analysis.SERS_Fitting.Auto_Fit_Raman as afr
start = time.time()
x = afr.Run(spectrum.y_smooth[fit_range[0]:fit_range[1]], wn_cal[fit_range[0]:fit_range[1]], Width = 0.01, Smoothing_Factor=0, Noise_Threshold = 0.01)
finish = time.time()
print(finish - start)

x = x[0].reshape(-1,3)

y = []
for i, this_peak in enumerate(x):
    height = x[i][0]
    mu = x[i][1]
    sigma = x[i][2]
    area =  height * sigma * (2*np.pi)**(1/2) 
    b = 0
    y.append(gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b))

y = np.array(y)
y_tot = np.sum(y, axis = 0)
# y_tot = y_tot - 3500
residuals = spectrum.y_smooth[fit_range[0]:fit_range[1]] - y_tot[fit_range[0]:fit_range[1]]
residuals_tot = np.sum(np.abs(residuals))

fig, ax = plt.subplots(1,1,figsize=[18,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
fig.suptitle('Auto Fit Raman()')
ax.plot(wn_cal, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
for i,peak in enumerate(x):
    ax.scatter(x[i][1], x[i][0], marker = 'x', s = 100)
    ax.plot(spectrum.x, y[i], zorder = 1, linestyle = 'dashed')
ax.plot(spectrum.x, y_tot, color = (1,0,0,0.6), linewidth = 2, label = 'Fit', zorder = 2)
ax.plot(spectrum.x[fit_range[0]:fit_range[1]], residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
ax.plot(1,1, label = 'Run time (s): ' + str(finish - start))
ax.set_xlim(spectrum.x[fit_range[0]],spectrum.x[fit_range[1]])
ax.legend()

#%% Test ifr

fit_range = [550,650]   
particle = particles[32]
powerseries = particle.powerseries
spectrum = powerseries[0]
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)


import nplab.analysis.SERS_Fitting.Iterative_Raman_Fitting as ifr
start = time.time()
x = ifr.Run(spectrum.x[fit_range[0]:fit_range[1]], Signal = spectrum.y_smooth[fit_range[0]:fit_range[1]], Regions=10, Peak_Type='G', Maximum_FWHM=50, Minimum_Width_Factor=0.1)#, Initial_Fit=initial_fit_new)
finish = time.time()
print(finish - start)

x = x[0].reshape(-1,3)

y = []
for i, this_peak in enumerate(x):
    height = x[i][0]
    mu = x[i][1]
    sigma = x[i][2]
    area =  height * sigma * (2*np.pi)**(1/2) 
    b = 0
    y.append(gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b))

y = np.array(y)
y_tot = np.sum(y, axis = 0)
# y_tot = y_tot - 3500
residuals = spectrum.y_smooth[fit_range[0]:fit_range[1]] - y_tot[fit_range[0]:fit_range[1]]
residuals_tot = np.sum(np.abs(residuals))

fig, ax = plt.subplots(1,1,figsize=[18,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
fig.suptitle('Iterative Fit Raman()')
ax.plot(wn_cal, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
for i,peak in enumerate(x):
    ax.scatter(x[i][1], x[i][0], marker = 'x', s = 100)
    ax.plot(spectrum.x, y[i], zorder = 1, linestyle = 'dashed')
ax.plot(spectrum.x, y_tot, color = (1,0,0,0.6), linewidth = 2, label = 'Fit', zorder = 2)
ax.plot(spectrum.x[fit_range[0]:fit_range[1]], residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
ax.plot(1,1, label = 'Run time (s): ' + str(finish - start))
ax.set_xlim(spectrum.x[fit_range[0]],spectrum.x[fit_range[1]])
ax.legend()


#%% Test ifr with scipy.find_peaks() as initial

fit_range = [550,650]   
particle = particles[32]
powerseries = particle.powerseries
spectrum = powerseries[0]
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)


# Find rough peak positions with scipy.find_peaks()

peak_fits, properties = scipy.signal.find_peaks(spectrum.y_smooth,
                                height=spectrum.y_smooth.max()/30,
                                threshold = None,
                                distance = 1,
                                prominence = 0.01,
                                width = 0.01,
                                wlen = None,
                                rel_height = 0.5,
                                plateau_size = None)

peaks = []
for i, peak in enumerate(peak_fits):
    mu = wn_cal[peak-1]
    width = properties['widths'][i]
    height = properties['peak_heights'][i]
    sigma = width/2.35
    area =  height * sigma * (2*np.pi)**(1/2)   
    # b = (spectrum.y_smooth[properties['right_bases'][i]] + spectrum.y_smooth[properties['left_bases'][i]])/2
    b = 0
    this_peak = Peak(mu = mu,
                     height = height,
                     width = width,
                     sigma = sigma, 
                     area = area)
    this_peak.x = wn_cal
    this_peak.y = gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b)
    
    peaks.append(this_peak)

## Load into initial fit array for later ifr fitting

initial_fit = []
for i, peak in enumerate(peaks):
    this_fit = np.zeros(3)
    this_fit[0] = peak.height
    this_fit[1] = peak.width
    this_fit[2] = peak.sigma
    initial_fit.append(peak.mu)

initial_fit = initial_fit[np.where(initial_fit >= spectrum.x[fit_range[0]])[0].min():np.where(initial_fit <= spectrum.x[fit_range[1]])[0].max()]


# IFR fitting

import nplab.analysis.SERS_Fitting.Iterative_Raman_Fitting as ifr
start = time.time()
x = ifr.Run(spectrum.x[fit_range[0]:fit_range[1]], Signal = spectrum.y_smooth[fit_range[0]:fit_range[1]], Regions=10, Peak_Type='G', Maximum_FWHM=50, Minimum_Width_Factor=0.1, Initial_Fit=initial_fit)
finish = time.time()
print(finish - start)

x = x[0].reshape(-1,3)

y = []
for i, this_peak in enumerate(x):
    height = x[i][0]
    mu = x[i][1]
    sigma = x[i][2]
    area =  height * sigma * (2*np.pi)**(1/2) 
    b = 0
    y.append(gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b))

y = np.array(y)
y_tot = np.sum(y, axis = 0)
# y_tot = y_tot - 3500
residuals = spectrum.y_smooth[fit_range[0]:fit_range[1]] - y_tot[fit_range[0]:fit_range[1]]
residuals_tot = np.sum(np.abs(residuals))

fig, ax = plt.subplots(1,1,figsize=[18,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
fig.suptitle('Iterative Fit Raman()')
ax.plot(wn_cal, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
for i,peak in enumerate(x):
    ax.scatter(x[i][1], x[i][0], marker = 'x', s = 100)
    ax.plot(spectrum.x, y[i], zorder = 1, linestyle = 'dashed')
ax.plot(spectrum.x, y_tot, color = (1,0,0,0.6), linewidth = 2, label = 'Fit', zorder = 2)
ax.plot(spectrum.x[fit_range[0]:fit_range[1]], residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
ax.plot(1,1, label = 'Run time (s): ' + str(finish - start))
ax.set_xlim(spectrum.x[fit_range[0]],spectrum.x[fit_range[1]])
ax.legend()

#%% Test spectrum_tools.approx_peak_gausses


def gaussian(x, height, center, width, width_height_frac = 0.5):
    a = height
    b = center
    c = width/(2*np.sqrt(2*np.log(1/width_height_frac)))
    
    return a*np.exp(-(((x - b)**2)/(2*c**2)))

fit_range = [None,None]   
particle = particles[32]
powerseries = particle.powerseries
spectrum = powerseries[0]
spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)

height_frac = 0.1
start = time.time()
x = spt.approx_peak_gausses(spectrum.x[fit_range[0]:fit_range[1]], spectrum.y_smooth[fit_range[0]:fit_range[1]], smooth_first=False, plot= False, height_frac = height_frac, threshold=0.1)
finish = time.time()
print(finish - start)

y  = []
for i, this_peak in enumerate(x):
    height = x[i][0]
    mu = x[i][1]
    width = x[i][2]
    sigma = width/2.35
    area =  height * sigma * (2*np.pi)**(1/2) 
    b = 0
    # y.append(gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b))
    y.append(gaussian(wn_cal, height, mu, width, height_frac))

y = np.array(y)
y_tot = y.sum(axis = 0)

residuals = spectrum.y_smooth[fit_range[0]:fit_range[1]] - y_tot[fit_range[0]:fit_range[1]]
residuals_tot = np.sum(np.abs(residuals))

fig, ax = plt.subplots(1,1,figsize=[18,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
fig.suptitle('spt.approx_peak_gausses()')
ax.plot(wn_cal, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
for i,peak in enumerate(x):
    ax.scatter(x[i][1], x[i][0], marker = 'x', s = 100)
    ax.plot(spectrum.x, y[i], zorder = 1, linestyle = 'dashed')
ax.plot(spectrum.x, y_tot, color = (1,0,0,0.6), linewidth = 2, label = 'Fit', zorder = 2)
ax.plot(spectrum.x[fit_range[0]:fit_range[1]], residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
ax.plot(1,1, label = 'Run time (s): ' + str(finish - start))
# ax.set_xlim(spectrum.x[fit_range[0]],spectrum.x[fit_range[1]])
ax.set_xlim(1350,1500)
ax.legend()


#%% Fit peaks for single powerseries - Quality Control


particle = particles[10]
powerseries = particle.powerseries

for j, spectrum in enumerate(powerseries):

    spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)
    print('\n')
    print(spectrum.name)
    height_frac = 0.1
    passed = False
    
    ## Loop through height_frac until fit residuals are acceptable
    while passed == False:
        start = time.time()
        x = spt.approx_peak_gausses(spectrum.x, spectrum.y_smooth, smooth_first=False, plot= False, height_frac = height_frac, threshold=0.15)
        finish = time.time()
        
        y  = []
        for i, this_peak in enumerate(x):
            height = x[i][0]
            mu = x[i][1]
            width = x[i][2]
            sigma = width/2.35
            area =  height * sigma * (2*np.pi)**(1/2) 
            b = 0
            # y.append(gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = b))
            y.append(gaussian(wn_cal, height, mu, width, height_frac))
        
        y = np.array(y)
        y_tot = y.sum(axis = 0)
        
        residuals = spectrum.y_smooth - y_tot
        residuals_tot = np.sum(np.abs(residuals))
    
        if np.abs(residuals).max() <= np.percentile(spectrum.y_smooth, 80) and residuals_tot <= np.sum(spectrum.y_smooth)*0.3:
            passed = True
            print('Passed')
        else:
            height_frac -= 0.01
            
        if height_frac <= 0.01:
            passed = True
            print('Failed')
            continue
    
    print(residuals_tot/np.sum(spectrum.y_smooth))
    print(np.abs(residuals).max())
    print(np.percentile(spectrum.y_smooth, 75))
    
    fig, ax = plt.subplots(1,1,figsize=[18,9])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
    ax.set_ylabel('SERS Intensity (cts/mW/s)')
    fig.suptitle(spectrum.name)
    ax.plot(wn_cal, spectrum.y_smooth, color = (0,0,0,0.5), linewidth = 1, label = 'Data')
    for i,peak in enumerate(x):
        ax.scatter(x[i][1], x[i][0], marker = 'x', s = 100)
        ax.plot(spectrum.x, y[i], zorder = 1, linestyle = 'dashed')
    ax.plot(spectrum.x, y_tot, color = (1,0,0,0.6), linewidth = 2, label = 'Fit', zorder = 2)
    ax.plot(spectrum.x, residuals, color = 'black', linewidth = 2, label = 'Residuals - Sum  = ' + str(np.round(residuals_tot,0)))
    ax.plot(1,1, label = 'Run time (s): ' + str(finish - start))
    ax.plot(1,1, label = 'Height frac: ' + str(height_frac))
    # ax.set_xlim(spectrum.x[fit_range[0]],spectrum.x[fit_range[1]])
    ax.set_xlim(1200,1600)
    ax.legend()

#%% Fit peaks for all particles

for j, particle in tqdm(enumerate(particles), leave = True):
    
    peaks = []
    
    powerseries = particle.powerseries
    
    for i, spectrum in enumerate(powerseries):
        spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 4000, fs = 30000)
        peak_fits, properties = scipy.signal.find_peaks(spectrum.y_smooth,
                                        height=spectrum.y_smooth.max()/10,
                                        threshold=None,
                                        distance=10,
                                        prominence=100,
                                        width=2,
                                        wlen=None,
                                        rel_height=0.5,
                                        plateau_size=None)
    
        these_peaks = []
        
        for i, peak in enumerate(peak_fits):
            mu = wn_cal[peak-1]
            width = properties['widths'][i]
            height = properties['peak_heights'][i]
            sigma = width/2.35
            area =  height * sigma * (2*np.pi)**(1/2)   
            this_peak = Peak(mu = mu,
                             height = height,
                             width = width,
                             sigma = sigma, 
                             area = area)
            this_peak.x = wn_cal
            this_peak.y = gauss(wn_cal, a = area, mu = mu, sigma = sigma, b = 0)
            
            these_peaks.append(this_peak)

        peaks.append(these_peaks)

    particle.peaks = peaks


#%% Get chosen peaks

for particle in particles:

    particle.peak_1280 = []
    particle.peak_1330 = []
    particle.peak_1420 = []    
    particle.peak_1620 = []
    
    for these_peaks in peaks:
        
        for peak in these_peaks:
            
            if peak.mu > 1260 and peak.mu < 1300:
                particle.peak_1280.append(peak)
            
            if peak.mu > 1310 and peak.mu < 1345:
                particle.peak_1330.append(peak)
                
            if peak.mu > 1400 and peak.mu < 1430:
                particle.peak_1420.append(peak)
            
            if peak.mu < 1650 and peak.mu > 1600:
                particle.peak_1620.append(peak)

    if len(particle.peak_1280) != 20:
        print(particle.name)
        print(len(particle.peak_1280))

        print('1280')

    if len(particle.peak_1330) != 20:
        print(particle.name)
        print(len(particle.peak_1330))
        print('1330')                

    if len(particle.peak_1420) != 20:
        print(particle.name)
        print(len(particle.peak_1420))
        print('1420')
        
    if len(particle.peak_1620) != 20:
        print(particle.name)
        print('1620')


#%% Get averages & errors of chosen peaks' amplitude

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
        this_avg_1280.append(particle.peak_1280[j].area)
    avg_1280.append(this_avg_1280)
avg_1280 = np.array(avg_1280)
mean_1280 = np.mean(avg_1280, axis = 0)
error_1280 = np.std(avg_1280, axis = 0)/sqrt(31)        

avg_1330 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1330 = []
    for j, spectrum in enumerate(particle.powerseries):
        this_avg_1330.append(particle.peak_1330[j].area)
    avg_1330.append(this_avg_1330)
avg_1330 = np.array(avg_1330)
mean_1330 = np.mean(avg_1330, axis = 0)
error_1330 = np.std(avg_1330, axis = 0)/sqrt(31)    

avg_1420 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1420 = []
    for j, spectrum in enumerate(particle.powerseries):
        this_avg_1420.append(particle.peak_1420[j].area)
    avg_1420.append(this_avg_1420)
avg_1420 = np.array(avg_1420)
mean_1420 = np.mean(avg_1420, axis = 0)
error_1420 = np.std(avg_1420, axis = 0)/sqrt(31)       

avg_1620 = []
for i, particle in enumerate(particles[0:31]):
    this_avg_1620 = []
    for j, spectrum in enumerate(particle.powerseries):
        this_avg_1620.append(particle.peak_1620[j].area)
    avg_1620.append(this_avg_1620)
avg_1620 = np.array(avg_1620)
mean_1620 = np.mean(avg_1620, axis = 0)
error_1620 = np.std(avg_1620, axis = 0)/sqrt(31)     
            
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
ax.errorbar(scan, mean_1420, yerr = error_1420, marker = 'o', markersize = 6, color = 'blue', linewidth = 1, label = '1420 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
ax.errorbar(scan, mean_1620, yerr = error_1620, marker = 'o', markersize = 6, color = 'purple', linewidth = 1, label = '1620 cm$^{-1}$', zorder = 2, markerfacecolor = 'none', markeredgewidth = 2, capsize = 5, elinewidth = 2)
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
