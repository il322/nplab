# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Co-TAPP-SMe 633nm SERS Kinetic Powerseries
(adapted from 2023-12-19_633nm_Powerswitch_Recovery_Plotter_v2.py)

- Updated Single, Double, & Triple Gaussian peak fitting
- Stores data in Particle() class
- Plot timescans with dark time, background timescans, fitted peak area & position of several peaks, 
  and recovery of peak area and background through kinetic scan
- Exponential decay fit of scan peak areas & background

  


Data: 2024-01-05_633nm_SERS_Powerseries_LongKinetic.h5


(samples:
     2023-11-28_Co-TAPP-SMe_60nm_MLAgg_b)

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
my_h5 = h5py.File(r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-01-05_633nm_SERS_Powerseries_LongKinetic.h5")


#%% Spectral calibration

# Using 2024-04-10.h5 data for cal, since no ref measurements in this dataset

cal_h5 = h5py.File(r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-04-10_633nm-BPT-MLAgg-Powerseries_633nm-Co-TAPP-SMe_MLAgg-Powerswitch_CCD-Flatness.h5")


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


#%% Compare calibration to 2024-04-10 spectra

stretch = 1.014
shift = -42

## 2024-01-05 Co-TAPP-SMe spectrum
spectrum01 = my_h5['ParticleScannerScan_0']['Particle_1']['SERS_1s_1.0uW_0']
spectrum01 = SERS.SERS_Spectrum(spectrum01)
spectrum01.x = spt.wl_to_wn(spectrum01.x, 632.8)
spectrum01.x = spectrum01.x + coarse_shift
spectrum01.x = spectrum01.x * coarse_stretch
spectrum01.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
spectrum01.x = wn_cal
spectrum01.y = np.sum(spectrum01.y[0:20], axis = 0)
spectrum01.normalise()

## 2024-04-10 Co-TAPP-SMe spectrum
spectrum = cal_h5['ParticleScannerScan_3']['Particle_2']['SERS_Powerseries_0']
spectrum = SERS.SERS_Spectrum(spectrum)
spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
spectrum.x = spectrum.x + coarse_shift
spectrum.x = spectrum.x * coarse_stretch
spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
spectrum.x = wn_cal
spectrum.normalise()
spectrum04 = spectrum

## Plot both
fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
ax.set_ylabel('SERS Intensity (cts/mW/s)')
ax.plot((spectrum01.x * stretch) + shift, spectrum01.y_norm, color = 'blue', label = 'Co-TAPP 01-05')
ax.plot(spectrum04.x, spectrum04.y_norm, color = 'red', label = 'Co-TAPP 04-10')
# ax.plot(bpt12.x, bpt12.y_norm+1, color = 'darkblue', label = 'BPT 12-19')
# ax.plot(bpt04.x, bpt04.y_norm+1, color = 'darkred', label = 'BPT 04-10', alpha = 0.3)
ax.legend()
# ax.set_xlim(460, 550)
# ax.set_xlim(750, 900)
# ax.set_xlim(1100, 1160)
ax.set_xlim(1150, 1240)
# ax.set_xlim(1200, 1450)
# ax.set_xlim(1610, 1660)
# ax.set_xlim(400, 1800)


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r"C:\Users\ishaa\OneDrive\Desktop\Offline Data\2024-01-05 633nm Kinetic Decay Powerseries Analysis NEW\\" + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%% Testing background subtraction & cosmic ray removal

dark_counts = 300

particle = my_h5['ParticleScannerScan_0']['Particle_0']



keys = list(particle.keys())
keys = natsort.natsorted(keys)
powerseries = []

for key in keys:
    if 'SERS' in key:
        timescan = particle[key]

chunk_size = 10
timescan = SERS.SERS_Timescan(timescan, exposure = timescan.attrs['cycle_time'])
timescan.x = spt.wl_to_wn(timescan.x, 632.8)
timescan.x = timescan.x + coarse_shift
timescan.x = timescan.x * coarse_stretch
timescan.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
timescan.x = wn_cal
timescan.x = timescan.x * stretch
timescan.x += shift
timescan.calibrate_intensity(R_setup = R_setup,
                              dark_counts = dark_counts,
                              exposure = timescan.exposure,
                              laser_power = timescan.laser_power)

timescan.chunk(chunk_size)
timescan.Y = timescan.Y/chunk_size
timescan.Baseline = deepcopy(timescan.Y)

for i, spectrum in tqdm(enumerate(timescan.Y), leave = True):
    spectrum = SERS.SERS_Spectrum(x = timescan.x, y = spectrum)
    spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
    spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    spectrum.y_baselined = spectrum.y - spectrum.baseline
    timescan.Y[i] = spectrum.y_baselined
    timescan.Baseline[i] = spectrum.baseline


fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
t_plot = np.linspace(0,len(timescan.Y)*timescan.exposure,len(timescan.Y))
v_min = timescan.Y.min()
v_max = np.percentile(timescan.Y, 99.9)
cmap = plt.get_cmap('inferno')
ax.set_yticklabels([])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
ax.set_ylabel('Time (s)', fontsize = 'large')
ax.set_yticks(np.linspace(0,len(timescan.Y)*timescan.exposure,11))
ax.set_yticklabels(np.linspace(0,len(timescan.Y)*timescan.exposure,11).astype('int'))
ax.set_title('633 nm SERS Timescan - Laser Power: ' + str(np.round(timescan.laser_power*1000,1)) + '$\mu$W\n' + str(particle.name), fontsize = 'x-large', pad = 10)
pcm = ax.pcolormesh(timescan.x, t_plot, timescan.Y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
clb = fig.colorbar(pcm, ax=ax)
clb.set_label(label = 'SERS Intensity', size = 'large', rotation = 270, labelpad=30)


fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
t_plot = np.linspace(0,len(timescan.Y)*timescan.exposure,len(timescan.Y))
v_min = timescan.Baseline.min()
v_max = np.percentile(timescan.Baseline, 99.9)
cmap = plt.get_cmap('inferno')
ax.set_yticklabels([])
ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
ax.set_ylabel('Time (s)', fontsize = 'large')
ax.set_yticks(np.linspace(0,len(timescan.Y)*timescan.exposure,11))
ax.set_yticklabels(np.linspace(0,len(timescan.Y)*timescan.exposure,11).astype('int'))
ax.set_title('Background 633 nm Timescan - Laser Power: ' + str(np.round(timescan.laser_power*1000,1)) + '$\mu$W\n' + str(particle.name), fontsize = 'x-large', pad = 10)
pcm = ax.pcolormesh(timescan.x, t_plot, timescan.Baseline, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
clb = fig.colorbar(pcm, ax=ax)
clb.set_label(label = 'Intensity', size = 'large', rotation = 270, labelpad=30)


#%% Get all particles to analyze into Particle class with h5 locations and in a list


particles = []

scan_list = ['ParticleScannerScan_0']

# Loop over particles in target particle scan

for particle_scan in scan_list:
    particle_list = []
    particle_list = natsort.natsorted(list(my_h5[particle_scan].keys()))
    
    ## Loop over particles in particle scan
    for particle in particle_list:
        if 'Particle_' not in particle:
            particle_list.remove(particle)
            
        if '97' in particle:
            particle_list.remove(particle)
           
            
    # Loop over particles in particle scan
    
    for particle in particle_list:
        
        ## Save to class and add to list
        this_particle = Particle()
        this_particle.name = 'MLAgg_' + str(particle_scan) + '_' + particle
        this_particle.h5_address = my_h5[particle_scan][particle]
        particles.append(this_particle)


#%% Functions to add & process SERS powerseries for each particle


def process_timescan(particle, chunk_size = 1):
    
    ## Get timescan object
    keys = list(particle.h5_address.keys())
    keys = natsort.natsorted(keys)
    for key in keys:
        if 'SERS' in key:
            timescan = particle.h5_address[key]
    
    ## Timescan calibration
    timescan = SERS.SERS_Timescan(timescan, exposure = timescan.attrs['cycle_time'])
    timescan.x = spt.wl_to_wn(timescan.x, 632.8)
    timescan.x = timescan.x + coarse_shift
    timescan.x = timescan.x * coarse_stretch
    timescan.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    timescan.x = wn_cal
    timescan.x = timescan.x * stretch
    timescan.x += shift
    timescan.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_counts,
                                  exposure = timescan.exposure,
                                  laser_power = timescan.laser_power)
    
    ## Chunk timescan for improved S/N
    timescan.chunk(chunk_size)
    timescan.chunk_size = chunk_size
    timescan.Y = timescan.Y/chunk_size
    timescan.Baseline = deepcopy(timescan.Y)
    timescan.BaselineSum = np.zeros(len(timescan.Y))
    timescan.peaks = []
    
    ## Background subtract each spectrum
    for i, spectrum in enumerate(timescan.Y):
        spectrum = SERS.SERS_Spectrum(x = timescan.x, y = spectrum)
        spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
        spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
        spectrum.y_baselined = spectrum.y - spectrum.baseline
        timescan.Y[i] = spectrum.y_baselined
        timescan.Baseline[i] = spectrum.baseline
        timescan.BaselineSum[i] = np.sum(timescan.Baseline[i])
        timescan.peaks.append([])
    
    particle.laser_power = timescan.laser_power
    particle.timescan = timescan
      

#%% Loop over all particles and process powerswitch

print('\nProcessing spectra...')

for i, particle in tqdm(enumerate(particles), leave = True):
    
    if i < 20:
        chunk_size = 20
    elif i < 40:
        chunk_size = 10
    elif i < 60:
        chunk_size = 4
    else:
        chunk_size = 2
    
    process_timescan(particle, chunk_size = chunk_size)

  
## Empirically tested chunk size for different powers
    ## Particle 0-19 = 10
    ## Particle 20-39 = 5
    ## Particle 40-59 = 2
    ## Particle 60-   = 1
#%% Peak fitting functions

def exp_decay(x, m, t, b):
    return m * np.exp(-t * x) + b

       
def gaussian(x, height, mu, width, baseline):
    
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    
    return height * np.exp(-(((x - mu)**2)/(2 * sigma **2))) + baseline


def gaussian2(x, 
              height1, mu1, width1, baseline1,
              height2, mu2, width2, baseline2):
    
    sigma1 = width1 / (2 * np.sqrt(2 * np.log(2)))
    sigma2 = width2 / (2 * np.sqrt(2 * np.log(2)))    
    
    y1 = height1 * np.exp(-(((x - mu1)**2)/(2 * sigma1 **2))) + baseline1
    y2 = height2 * np.exp(-(((x - mu2)**2)/(2 * sigma2 **2))) + baseline2
    
    return y1 + y2


def gaussian3(x, 
              height1, mu1, width1, baseline1,
              height2, mu2, width2, baseline2,
              height3, mu3, width3, baseline3):
    
    sigma1 = width1 / (2 * np.sqrt(2 * np.log(2)))
    sigma2 = width2 / (2 * np.sqrt(2 * np.log(2)))    
    sigma3 = width3 / (2 * np.sqrt(2 * np.log(2)))
    
    y1 = height1 * np.exp(-(((x - mu1)**2)/(2 * sigma1 **2))) + baseline1
    y2 = height2 * np.exp(-(((x - mu2)**2)/(2 * sigma2 **2))) + baseline2
    y3 = height3 * np.exp(-(((x - mu3)**2)/(2 * sigma3 **2))) + baseline3
    
    return y1 + y2 + y3
    

def lorentzian(x, height, mu, fwhm):
    I = height
    x0 = mu
    gamma = fwhm/2
    numerator = gamma**2
    denominator = (x - x0)**2 + gamma**2
    quot = numerator/denominator
    
    y = I*quot
    return y


#%% Peak fit for single Gaussian regions  
        

def fit_gaussian_timescan(particle, fit_range, peak_name = None, R_sq_thresh = 0.85, smooth_first = False, plot = False):

    '''
    Fit single gaussian peak to given range of each spectrum in timescan
    Adds peaks to timescan.peaks[i]
    '''    
    
    timescan = particle.timescan

    for i, spectrum in enumerate(timescan.Y):
        
    
        ## Get region of spectrum to fit
        fit_range_index = [np.abs(timescan.x-fit_range[0]).argmin(), np.abs(timescan.x-fit_range[1]).argmin()+1]
        x = timescan.x[fit_range_index[0]:fit_range_index[1]]
        y = spectrum[fit_range_index[0]:fit_range_index[1]]
       
        if smooth_first == True:
            spectrum_smooth = spt.butter_lowpass_filt_filt(spectrum, cutoff = 3000, fs = 30000)
            y = spectrum_smooth[fit_range_index[0]:fit_range_index[1]]
        
    
        # Fit
        
        ## Initial guesses
        i_max = y.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height0 = y[i_max]
        mu0 = x[i_max] # centre x position of guessed peak
        baseline0 = (y[0] + y[len(y)-1])/2 # height of baseline guess
        i_half = np.argmax(y >= (height0 + baseline0)/2) # Index of first argument to be at least halfway up the estimated bell
        # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
        # half, and even if this point is a distant outlier the fit should still converge.
        sigma0 = (mu0 - x[i_half]) / np.sqrt(2*np.log((height0 - baseline0)/(y[i_half] - baseline0)))
        width0 = sigma0 * 2 * np.sqrt(2*np.log(2))
        # sigma0 = 3
        p0 = height0, mu0, width0, baseline0
        
        ## Perform fit (height, mu, width, baseline)
        start = time.time()
        
        try:
            popt, pcov = curve_fit(f = gaussian, 
                                xdata = x,
                                ydata = y, 
                                p0 = p0,
                                bounds=(
                                    (0, x.min(), 0, y.min()),
                                    (height0 * 1.2, x.max(), (x.max() - x.min()) * 2, height0),),)
            
        except:
            popt = [height0, mu0, sigma0, baseline0]
            # print('\nFit Error')
          
        finish = time.time()
        runtime = finish - start
            
        # print('Guess:', np.array(p0))
        # print('Fit:  ', popt)
        
        ## Get fit data
        height = popt[0]
        mu = popt[1]
        width = popt[2]
        baseline = popt[3]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt)
        residuals = y - y_fit
        residuals_sum = np.abs(residuals).sum()
        R_sq =  1 - ((np.sum(residuals**2)) / (np.sum((y - np.mean(y))**2)))
        
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        
        ## Screening poor fits
        if R_sq < R_sq_thresh or np.isnan(R_sq):
            # print('\nPoor Fit')
            # print(particle.name)
            # print(spectrum.name)
            # print('R^2 = ' + str(np.round(R_sq, 3)))
            # print('Guess:', np.array(p0))
            # print('Fit:  ', popt)
            error = True
        else:
            error = False
        
        ## Add fit data to peak class
        this_peak = Peak(*popt)
        this_peak.area = area
        this_peak.sigma = sigma
        this_peak.name = peak_name
        this_peak.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        this_peak.R_sq = R_sq
        this_peak.residuals = SERS.SERS_Spectrum(x = x, y = residuals)
        this_peak.error = error        
        
        ## Add peak class to timescan class 'peaks' list
        particle.timescan.peaks[i].append(this_peak)
           
        
        # Plot
        
        if plot == True: # or error == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title('Scan ' + str(i), fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
                ax.plot(x, y, color = 'grey', label = 'Smoothed') 
            else:
                ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
            ax.plot(x, y_fit, label = 'Fit', color = 'orange')
            ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
            ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
            ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
            ax.set_xlim(fit_range[0], fit_range[1])
            ax.set_ylim(None, y.max()*1.2)
            ax.legend()
            plt.show()


# Testing fit ranges

particle = particles[51]
# fit_gaussian_timescan(particle = particle, fit_range = [480, 545], smooth_first = False, plot = True)
# fit_gaussian_timescan(particle = particle, fit_range = [1100, 1150], smooth_first = False, plot = True)
# fit_gaussian_timescan(particle = particle, fit_range = [1170, 1220], smooth_first = False, plot = True)
# fit_gaussian_timescan(particle = particle, fit_range = [1280, 1310], smooth_first = False, plot = True)
# fit_gaussian_timescan(particle = particle, fit_range = [1628, 1655], smooth_first = False,  plot = True)    


#%% Peak fit for triple Gaussian region

def fit_gaussian3_timescan(particle, fit_range = [1365, 1450], peak_name = None, R_sq_thresh = 0.85, smooth_first = False, plot = False, save = False):
    
    '''
    Fit triple gaussian peak to given range of each spectrum in timescan (highly tuned for 1365 - 1450cm-1 region)
    Adds peaks to timescan.peaks[i]
    '''       
    
    timescan = particle.timescan

    for i, spectrum in enumerate(timescan.Y):
        
    
        ## Get region of spectrum to fit
        fit_range_index = [np.abs(timescan.x-fit_range[0]).argmin(), np.abs(timescan.x-fit_range[1]).argmin()+1]
        x = timescan.x[fit_range_index[0]:fit_range_index[1]]
        y = spectrum[fit_range_index[0]:fit_range_index[1]]
       
        if smooth_first == True:
            spectrum_smooth = spt.butter_lowpass_filt_filt(spectrum, cutoff = 3000, fs = 30000)
            y = spectrum_smooth[fit_range_index[0]:fit_range_index[1]]

        # return x

        # Fit
        
        ## Initial guesses

        ### First peak
        x1 = x[:21]
        y1 = y[:21]
        i_max = y1.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height1 = y1[i_max]
        mu1 = x1[i_max] # centre x position of guessed peak
        width1 = (x1.max()-x1.min())/2
        baseline1 = 0
        
        ### Second peak
        x2 = x[22:28]
        y2 = y[22:28]
        i_max = y2.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height2 = y2[i_max]
        mu2 = x2[i_max] # centre x position of guessed peak
        width2 = 14
        baseline2 = 0
        
        ### Third peak
        x3 = x[31:]
        y3 = y[31:]
        i_max = y3.argmax() # index of highest value - for guess, assumed to be Gaussian peak
        height3 = y3[i_max]
        mu3 = x3[i_max] # centre x position of guessed peak
        width3 = 13
        baseline3 = 0
        # baseline3 = (y3[0] + y3[len(y3)-1])/2 # height of baseline guess
        # i_half = np.argmax(3 >= (height3 + baseline3)/2) # Index of first argument to be at least halfway up the estimated bell
        # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
        # half, and even if this point is a distant outlier the fit should still converge.
        # sigma3 = (mu3 - x3[i_half]) / np.sqrt(2*np.log((height3 - baseline3)/(y3[i_half] - baseline3)))
        # width3 = sigma3 * 2 * np.sqrt(2*np.log(2))
        # sigma0 = 3
        # p0 = height0, mu0, width0, baseline0
        
        ## Perform fit (height, mu, width, baseline)
        start = time.time()
        
        ## Initial guesses
        
        ### First peak
        # height1 = y.max()/3
        # mu1 = 1399
        # width1 = 20
        # baseline1 = 0
        
        # ### Second peak
        # height2 = y.max()/1.7
        # mu2 = 1425
        # width2 = 15
        # baseline2 = 0
        
        # # ### Third peak
        # height3 = y.max()
        # mu3 = 1434
        # width3 = 15
        # baseline3 = 0
        
        p0 = [
                height1, mu1, width1, baseline1,
                height2, mu2, width2, baseline2,
                height3, mu3, width3, baseline3
             ]
        
        # lower_bounds = (
        #                 0, mu1 - 5, width1 - 5, y.min(),
        #                 0, mu2 - 5, width2 - 1, y.min(),
        #                 0, mu3 - 2, width3 - 1, y.min()
        #                 )
        
        # upper_bounds = (
        #                 height1 * 2, mu1 + 5, width1 + 2, 100,
        #                 height2 * 2, mu2 + 5, width2 + 1, 100,
        #                 height3 * 2, mu3 + 2, width3 + 1, 100
        #                 )

        lower_bounds = (
                        0, x1.min(), 0, y.min(),
                        height2 * 0.9, mu2 - 2, width2 - 5, y.min(),
                        height3 * 0.9, mu3 - 2, width3 - 5, y.min()
                        )
        
        upper_bounds = (
                        height2, x1.max(), x.max()-x.min(), 100,
                        height2 * 1.05, mu2 + 2, width2 + 5, 100,
                        height3 * 1.05, mu3 + 2, width3 + 5, 100
                        )        
        
        try:
            popt, pcov = curve_fit(f = gaussian3, 
                                xdata = x,
                                ydata = y, 
                                p0 = p0,
                                bounds=((lower_bounds),(upper_bounds),),)
            
        except:
            popt = p0
            # print('\nFit Error')
          
        finish = time.time()
        runtime = finish - start
        # print()
        # print('\n')
        # print(particle.name)
        # print(spectrum.name)
        # # print('R^2 = ' + str(np.round(R_sq, 3)))
        # print('Guess:', np.array(p0))
        # print('Fit:  ', popt)
        # print('Guess:', np.array(p0))
        # print('Fit:  ', popt)
        
        
        # Get fit data
        
        ## Peak 1
        height = popt[0]
        mu = popt[1]
        width = popt[2]
        baseline = popt[3]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[0:4])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak1 = Peak(*popt[0:4])
        peak1.area = area
        peak1.sigma = sigma
        peak1.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak1.name = '1400'

        ## Peak 2
        height = popt[4]
        mu = popt[5]
        width = popt[6]
        baseline = popt[7]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[4:8])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak2 = Peak(*popt[4:8])
        peak2.area = area
        peak2.sigma = sigma
        peak2.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak2.name = '1420'

        ## Peak 3
        height = popt[8]
        mu = popt[9]
        width = popt[10]
        baseline = popt[11]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        area = height * sigma * np.sqrt(2 * np.pi)
        y_fit = gaussian(x, *popt[8:12])
        if peak_name is None:
            peak_name = str(int(np.round(np.mean(fit_range),0)))
        peak3 = Peak(*popt[8:12])
        peak3.area = area
        peak3.sigma = sigma
        peak3.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
        peak3.name = '1435'
        
        ## Sum & residuals
        y_fit = peak1.spectrum.y + peak2.spectrum.y + peak3.spectrum.y
        residuals = y - y_fit
        residuals_sum = np.abs(residuals).sum()
        R_sq =  1 - ((np.sum(residuals**2)) / (np.sum((y - np.mean(y))**2)))
        ### R_sq is total R_sq for triple
        peak1.R_sq = R_sq
        peak2.R_sq = R_sq
        peak3.R_sq = R_sq
        
        # if peak_name is None:
        #     peak_name = str(int(np.round(np.mean(fit_range),0)))
        
        ## Screening poor fits
        if R_sq < R_sq_thresh or np.isnan(R_sq):
            # print('\nPoor Fit')
            # print(particle.name)
            # print(spectrum.name)
            # print('R^2 = ' + str(np.round(R_sq, 3)))
            # print('Guess:', np.array(p0))
            # print('Fit:  ', popt)
            error = True
        else:
            error = False
        peak1.error = error
        peak2.error = error
        peak3.error = error     
        
        ## Add peaks class to timescan class 'peaks' list
        particle.timescan.peaks[i].append(peak1)
        particle.timescan.peaks[i].append(peak2)        
        particle.timescan.peaks[i].append(peak3)

        
        # Plot
        
        if plot == True: # or error == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title('Scan ' + str(i), fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
                ax.plot(x, y, color = 'grey', label = 'Smoothed') 
            else:
                ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
            ax.plot(x, peak1.spectrum.y, label = 'Fit A', color = 'red')
            ax.plot(x, peak2.spectrum.y, label = 'Fit B', color = 'green')
            ax.plot(x, peak3.spectrum.y, label = 'Fit C', color = 'blue')
            ax.plot(x, y_fit, label = 'Fit', color = 'orange')
            ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
            ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
            ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
            ax.set_xlim(fit_range[0], fit_range[1])
            ax.set_ylim(None, y.max()*1.2)
            ax.legend()
            plt.show()
            
        ## Save
        if save == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title('Scan ' + str(i), fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
                ax.plot(x, y, color = 'grey', label = 'Smoothed') 
            else:
                ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
            ax.plot(x, peak1.spectrum.y, label = 'Fit A', color = 'red')
            ax.plot(x, peak2.spectrum.y, label = 'Fit B', color = 'green')
            ax.plot(x, peak3.spectrum.y, label = 'Fit C', color = 'blue')
            ax.plot(x, y_fit, label = 'Fit', color = 'orange')
            ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
            ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
            ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
            ax.set_xlim(fit_range[0], fit_range[1])
            ax.set_ylim(None, y.max()*1.2)
            ax.legend()
            
            save_dir = get_directory(particle.name + '\\Triple Fit')
            fig.savefig(save_dir + particle.name + '_' + 'Scan_' + str(i) + '_TripleFit' + '.svg', format = 'svg')
            plt.close(fig)


# Testing fit ranges

particle = particles[51]
# for spectrum in particle.powerseries:
#     spectrum.peaks = []
# fit_gaussian_powerseries(particle = particle, fit_range = [480, 545], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1100, 1150], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1170, 1220], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1280, 1310], smooth_first = True, plot = True)
# fit_gaussian3_timescan(particle = particle, fit_range = [1365, 1450], smooth_first = False, plot = True, save = False)
# fit_gaussian_powerseries(particle = particle, fit_range = [1475, 1525], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1628, 1655], smooth_first = True,  plot = True)    
# for spectrum in particle.powerseries:
#     try:
#         baseline = spectrum.peaks[6].baseline/spectrum.peaks[5].baseline
#     except:
#         baseline = 0
#     ratio = Peak(height = spectrum.peaks[6].height/spectrum.peaks[5].height,
#                  mu = None,
#                  width = spectrum.peaks[6].width/spectrum.peaks[5].width,
#                  baseline = baseline)
#     ratio.area = spectrum.peaks[6].area/spectrum.peaks[5].area
#     ratio.sigma = spectrum.peaks[6].sigma/spectrum.peaks[5].sigma
#     ratio.name = '1435/1420'
#     if spectrum.peaks[6].error or spectrum.peaks[5].error:
#         ratio.error = True
#     else:
#         ratio.error = False
#     spectrum.peaks.append(ratio)

# plot_peak_areas(particle, save = False)


#%% Loop over all particles and fit

print('\nPeak Fitting...')

for particle in tqdm(particles, leave = True):
    
    ## Clear previous fitted peaks
    for i, spectrum in enumerate(particle.timescan.Y):
        particle.timescan.peaks[i] = []
        
    ## Fit peaks
    fit_gaussian_timescan(particle = particle, fit_range = [480, 545], smooth_first = False, plot = False)
    fit_gaussian_timescan(particle = particle, fit_range = [1100, 1150], smooth_first = False, plot = False)
    fit_gaussian_timescan(particle = particle, fit_range = [1170, 1220], smooth_first = False, plot = False)
    fit_gaussian_timescan(particle = particle, fit_range = [1280, 1310], smooth_first = False, plot = False)
    fit_gaussian3_timescan(particle = particle, fit_range = [1365, 1450], smooth_first = False, plot = False, save = False)
    # fit_gaussian_powerseries(particle = particle, fit_range = [1475, 1525], smooth_first = True, plot = False)
    fit_gaussian_timescan(particle = particle, fit_range = [1628, 1655], smooth_first = False,  plot = False)    

    ## 1435/1425 ratio peak    
    for i in range(0, len(particle.timescan.Y)):
        peaks = particle.timescan.peaks[i]
        try:
            baseline = peaks[6].baseline/peaks[5].baseline
        except:
            baseline = 0
        ratio = Peak(height = peaks[6].height/peaks[5].height,
                     mu = None,
                     width = peaks[6].width/peaks[5].width,
                     baseline = baseline)
        ratio.area = peaks[6].area/peaks[5].area
        ratio.sigma = peaks[6].sigma/peaks[5].sigma
        ratio.name = '1435/1420'
        if peaks[6].error or peaks[5].error:
            ratio.error = True
        else:
            ratio.error = False
        peaks.append(ratio)


# Report number of failed fits

count = 0
scans = 0
for particle in particles:
    for i in range(0, len(particle.timescan.Y)):
        scans += 1
        for peak in particle.timescan.peaks[i]:
            if peak.error: count += 1
print('\nFit Errors (%): ')
print(100*count/ (len(particles) * scans * len(particle.timescan.peaks[0])))
  
#%% Plotting functions

    
# Plotting prep
my_cmap = plt.get_cmap('inferno')


def plot_timescan(particle, save = False):
    
    timescan = particle.timescan
    
    
    fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.linspace(0,len(timescan.Y)*timescan.exposure,len(timescan.Y))
    v_min = timescan.Y.min()
    v_max = np.percentile(timescan.Y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    ax.set_ylabel('Time (s)', fontsize = 'large')
    ax.set_yticks(np.linspace(0,len(timescan.Y)*timescan.exposure,11))
    ax.set_yticklabels(np.linspace(0,len(timescan.Y)*timescan.exposure,11).astype('int'))
    ax.set_title('633 nm SERS Timescan - Laser Power: ' + str(np.round(timescan.laser_power*1000,1)) + '$\mu$W\n' + str(particle.name), fontsize = 'x-large', pad = 10)
    pcm = ax.pcolormesh(timescan.x, t_plot, timescan.Y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax=ax)
    clb.set_label(label = 'SERS Intensity', size = 'large', rotation = 270, labelpad=30)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_633nm SERS Timescan' + '.svg', format = 'svg')
        plt.close(fig)


def plot_timescan_background(particle, save = False):
    
    timescan = particle.timescan
    
    fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.linspace(0,len(timescan.Y)*timescan.exposure,len(timescan.Y))
    v_min = timescan.Baseline.min()
    v_max = np.percentile(timescan.Baseline, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    ax.set_ylabel('Time (s)', fontsize = 'large')
    ax.set_yticks(np.linspace(0,len(timescan.Y)*timescan.exposure,11))
    ax.set_yticklabels(np.linspace(0,len(timescan.Y)*timescan.exposure,11).astype('int'))
    ax.set_title('Background 633 nm Timescan - Laser Power: ' + str(np.round(timescan.laser_power*1000,1)) + '$\mu$W\n' + str(particle.name), fontsize = 'x-large', pad = 10)
    pcm = ax.pcolormesh(timescan.x, t_plot, timescan.Baseline, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax=ax)
    clb.set_label(label = 'Intensity', size = 'large', rotation = 270, labelpad=30)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_633nm Background Timescan' + '.svg', format = 'svg')
        plt.close(fig)
        
        
def plot_peak_areas(particle, save = False):
       
    timescan = particle.timescan
    
    time = np.arange(0, len(timescan.Y) * timescan.chunk_size, timescan.chunk_size, dtype = int)   
    
    peaks_list = np.array([0,1,2,3,5,6,7,8])
    
    colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'deeppink', 'black']
    
    fig, axes = plt.subplots(len(peaks_list) + 1,1,figsize=[14,22], sharex = True)
    fig.suptitle('633 nm SERS Timescan - Laser Power: ' + str(np.round(timescan.laser_power*1000,1)) + '$\mu$W\n' + str(particle.name), fontsize = 'x-large')
    axes[len(axes)-1].set_xlabel('Time (s)', size = 'x-large')
    
    # axes[int(len(axes)/2)].set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
    
    peaks = timescan.peaks
    for i, peak in enumerate(timescan.peaks[0]):
        
        if i in peaks_list:

            ## Peak area ratios to plot for y
            y = [] 
            ax = axes[np.where(peaks_list == i)[0][0]]    
            ax.set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
                        
            for j in range(0, len(timescan.Y)):
                if peaks[j][i].error:
                    y.append(nan)
                elif peaks[j][i].name == '1435/1420':
                    y.append(peaks[j][i].area)
                else:
                    y.append(peaks[j][i].area/timescan.peaks[0][i].area)          
            y = np.array(y)
            peak_spec = SERS.SERS_Spectrum(x = time, y = y)
                                  
            ## Plot peaks
            color = colors[np.where(peaks_list == i)[0][0]]
            if 'Avg' in particle.name:
                alpha = 0.5
            else:
                alpha = 1
            ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 4, linewidth = 2, alpha = alpha)
            
            # ## ylim
            # if peaks[j][i].name == '1435/1420':
            #     pass
            # # else:
            # #     try:
            # #         ax.set_ylim(np.nanpercentile(peak_spec.y, 0.5), 1.05)
            # #     except: pass
            if 'Avg' in particle.name:
                
                fit_x = peak_spec.x
                fit_y = exp_decay(fit_x,
                                  particle.ms[i],
                                  particle.ts[i],
                                  particle.bs[i])
                ax.plot(fit_x, fit_y, color = color, zorder = 4, linewidth = 3, alpha = 1)
                
       
            # # Errorbars, if avg particle
            # try:                
            #     y_error = []
            #     for j in range(0, len(timescan.Y)):
            #         if peaks[j][i].error:
            #             y_error.append(nan)
            #         elif peaks[j][i].name == '1435/1420':
            #             y_error.append(peaks[j][i].area_error)
            #         else:
            #             this_error = (peaks[j][i].area/peaks[j][0].area) * np.sqrt((peaks[j][i].area_error/peaks[j][i].area)**2 + (peaks[j][0].area_error/peaks[j][0].area)**2)
                        
            #             y_error.append(this_error)   
            #     y_error = np.array(y_error)
            #     ax.errorbar(peak_spec.x, peak_spec.y, yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
            # except:
            #     y_error = None
            
                
        ## Plot background sum
        y = []
        for j in range(0, len(timescan.Y)):
            y.append(timescan.BaselineSum[j])
        y = np.array(y)
        background_spec = SERS.SERS_Spectrum(x = time, y = y)                             
        color = colors[len(colors)-1]
        axes[len(axes)-1].plot(background_spec.x, background_spec.y, label = 'Background Sum', color = color, alpha = 0.1, zorder = 4, linewidth = 2)
        if 'Avg' in particle.name:
            
            fit_x = background_spec.x
            fit_y = exp_decay(fit_x,
                              particle.ms[-1],
                              particle.ts[-1],
                              particle.bs[-1])
            axes[len(axes)-1].plot(fit_x, fit_y, color = color, zorder = 4, linewidth = 3, alpha = 0.5)


        ax.legend(loc = 'upper right')
    
        ylim = ax.get_ylim()            
        
    
    axes[len(axes)-2].set_ylabel('I$_{1435}$ / I$_{1420}$', size = 'x-large')
    axes[len(axes)-1].set_ylabel('Background', size = 'large')
    plt.tight_layout(pad = 1.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_Peak Area' + '.svg', format = 'svg')
        plt.close(fig)
        
        
def plot_peak_positions(particle, save = False):
           
    timescan = particle.timescan
    
    time = np.arange(0, len(timescan.Y) * timescan.chunk_size, timescan.chunk_size, dtype = int)   
    
    peaks_list = np.array([0,1,2,3,5,6,7,8])
    
    colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'deeppink', 'black']
    
    fig, axes = plt.subplots(len(peaks_list) - 1,1,figsize=[14,24], sharex = True)
    fig.suptitle('633 nm SERS Timescan - Laser Power: ' + str(np.round(timescan.laser_power*1000,1)) + '$\mu$W\n' + str(particle.name), fontsize = 'x-large')
    axes[len(axes)-1].set_xlabel('Time (s)', size = 'x-large')
    axes[int(len(axes)/2)].set_ylabel('Peak Position (cm$^{-1}$)', size = 'xx-large')
    
    # axes[int(len(axes)/2)].set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
    
    peaks = timescan.peaks
    for i, peak in enumerate(timescan.peaks[0][:-1]):
        
        if i in peaks_list:

            ## Peak area ratios to plot for y
            y = [] 
            ax = axes[np.where(peaks_list == i)[0][0]]    
                        
            for j in range(0, len(timescan.Y)):
                if peaks[j][i].error:
                    y.append(nan)
                else:
                    y.append(peaks[j][i].mu)          
            y = np.array(y)
            peak_spec = SERS.SERS_Spectrum(x = time, y = y)
                                  
            ## Plot peaks
            color = colors[np.where(peaks_list == i)[0][0]]
            ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 4, linewidth = 2)
       
            # ## Errorbars, if avg particle
            # try:                
            #     y_error = []
            #     for j in range(0, len(timescan.Y)):
            #         if peaks[j][i].error:
            #             y_error.append(nan)
            #         else:
            #             this_error = peaks[j][i].mu_error
            #             y_error.append(this_error)   
                        
            #     y_error = np.array(y_error)
            #     ax.errorbar(peak_spec.x, peak_spec.y, yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
            # except:
            #     y_error = None

            # ax.legend(loc = 'upper right')
        
            # ylim = ax.get_ylim()            
            # ax.set_ylim(ylim)
    
    # axes[len(axes)-1].set_ylabel('Background', size = 'large')
    plt.tight_layout(pad = 1.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_Peak Position' + '.svg', format = 'svg')
        plt.close(fig)


def plot_baseline_sum(particle, save = False):
    
    
    powerseries = particle.powerseries
    
    scan = np.arange(0, len(powerseries), 1, dtype = int)   
    
    fig, ax = plt.subplots(1, 1, figsize=[18,8], sharex = True)
    fig.suptitle('Background Sum 633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W' + '\n' + str(particle.name) + '\n' + 'Dark Time  = ' + str(np.round(particle.dark_time,2)) + 's', fontsize = 'x-large')
    ax.set_xlabel('Scan No.', size = 'x-large')
    ax.set_xticks(scan[::2])
    ax.set_xticklabels(scan[::2], fontsize = 'large')
    ax.set_ylabel('$\Sigma$  Background  Intensity', size = 'large')
    
    ## Background sum to plot for y
    y = []                
    for spectrum in powerseries:
        y.append(spectrum.baseline_sum)          
    y = np.array(y)
    
    baseline_spec = SERS.SERS_Spectrum(x = scan, y = y)
                          
    ## Plot
    color = 'black'
    ax.plot(baseline_spec.x, baseline_spec.y, color = color, zorder = 4)
    ax.scatter(baseline_spec.x[1::2], baseline_spec.y[1::2], label = '90 $\mu$W', marker = 'o', facecolors = color, edgecolors = color, linewidth = 3, s = 200, zorder = 2)
    ax.scatter(baseline_spec.x[::2], baseline_spec.y[::2], label = '1 $\mu$W', marker = 'o', facecolors = 'white', edgecolors = color, linewidth = 3, s = 200, zorder = 2)
   
    # Errorbars, if avg particle
    try:                
        y_error = []
        for spectrum in powerseries:                   
                y_error.append(spectrum.baseline_sum_error)   
        y_error = np.array(y_error)
        
        ax.errorbar(baseline_spec.x[1::2], baseline_spec.y[1::2], yerr = y_error[1::2], marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 10, elinewidth = 5, capthick = 3, ecolor = color, zorder = 1)
        ax.errorbar(baseline_spec.x[::2], baseline_spec.y[::2], yerr = y_error[::2], marker = 'none', mfc = 'white', mec = color, linewidth = 0, markersize = 10, capsize = 10, elinewidth = 5, capthick = 3, ecolor = color, zorder = 1)
    except:
        y_error = None

    ax.legend(loc = 'upper right', fontsize = 'large')

    ylim = ax.get_ylim()            
    if particle.dark_time > 0:
        ax.vlines(9.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
        # ax.vlines(19.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
        # ax.vlines(29.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
        ax.set_ylim(ylim)
    
    plt.tight_layout(pad = 0.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_Background Sum' + '.svg', format = 'svg')
        plt.close(fig)


# Test plotting

# particle = particles[0]
# plot_timescan(particle, save = False)
# plot_peak_areas(particle, save = False)
# plot_peak_positions(particle, save = False)
        
particle = particles[0]
# plot_timescan(particle, save = False)
plot_peak_areas(particle, save = False)
# plot_peak_positions(particle, save = False)


#%% Loop over all particles and plot

print('\nPlotting...')

for particle in tqdm(particles, leave = True):
    
    plot_timescan(particle, save = True)
    plot_timescan_background(particle, save = True)
    plot_peak_areas(particle, save = True)
    plot_peak_positions(particle, save = True)
 

#%% Making average particles & calculating timescan

from copy import deepcopy

avg_particles = []
laser_powers = []

## Get dark times
for particle in particles:
    if particle.laser_power not in laser_powers:
        laser_powers.append(particle.laser_power)


# Loop over each dark time - one avg particle per dark time
for i, laser_power in enumerate(laser_powers):
    ## Set up avg particle
    avg_particle = Particle()
    avg_particle.laser_power = laser_power
    avg_particle.name = 'MLAgg_Avg_' + str(np.round(laser_power*1000,1)) + 'uW'
    
    ## Set up avg timescan
    keys = list(particles[i*10].h5_address.keys())
    keys = natsort.natsorted(keys)
    for key in keys:
        if 'SERS' in key:
            timescan = particles[i*10].h5_address[key]
    timescan = SERS.SERS_Timescan(timescan, exposure = timescan.attrs['cycle_time'] * particles[i*10].timescan.chunk_size)
    timescan.x = spt.wl_to_wn(timescan.x, 632.8)
    timescan.x = timescan.x + coarse_shift
    timescan.x = timescan.x * coarse_stretch
    timescan.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    timescan.x = wn_cal
    timescan.x = timescan.x * stretch
    timescan.x += shift
    avg_particle.timescan = timescan
    
    avg_particle.timescan.Y = np.zeros(particles[i*10].timescan.Y.shape)
    avg_particle.timescan.Baseline = np.zeros(particles[i*10].timescan.Baseline.shape)
    # avg_particle.timescan.BaselineSum = np.zeros(particles[0].timescan.BaselineSum.shape)
    
    ## Add y-values to avg timescan
    counter = 0
    for particle in particles:
        if particle.laser_power == avg_particle.laser_power:
            counter += 1
            avg_particle.timescan.Y += particle.timescan.Y
            avg_particle.timescan.Baseline += particle.timescan.Baseline
            # avg_particle.timescan.BaselineSum += particle.timescan.BaselineSum
            avg_particle.timescan.chunk_size = particle.timescan.chunk_size          
            
    ## Divide
    avg_particle.timescan.Y = avg_particle.timescan.Y/counter
    avg_particle.timescan.Baseline = avg_particle.timescan.Baseline/counter
    # avg_particle.timescan.BaselineSum = avg_particle.timescan.BaselineSum/counter
                    
    avg_particles.append(avg_particle)
 
                      
#%% Get peak data for average particles

from copy import deepcopy

print('\nAveraging...')

for avg_particle in tqdm(avg_particles, leave = True):
    
    timescan = avg_particle.timescan
    
    ## Clear previous fitted peaks
    timescan.peaks = []
    for i, spectrum in enumerate(timescan.Y):
        timescan.peaks.append([])
    
    
    ## Set up peaks list
    for i, spectrum in enumerate(timescan.Y):
        
        timescan.BaselineSum = []
        timescan.BaselineSumError = []
        # spectrum.baseline_sum_recovery = []
        
        timescan.peaks[i] = deepcopy(particles[0].timescan.peaks[0])
        
        for peak in timescan.peaks[i]:
            peak.area = []
            peak.baseline = []
            peak.error = False
            peak.height = []
            peak.mu = []
            peak.sigma = []
            peak.width = []
    
    ## Add peaks
    counter = 0
    for particle in particles:
        if avg_particle.laser_power == particle.laser_power:
            counter += 1
            
            for i, spectrum in enumerate(timescan.Y):
                
                timescan.BaselineSum.append(particle.timescan.BaselineSum[i])
                timescan.BaselineSumError.append(0)
                
                for j, peak in enumerate(timescan.peaks[i]):
                    this_peak = deepcopy(particle.timescan.peaks[i][j])
                    peak.area.append(this_peak.area)
                    peak.baseline.append(this_peak.baseline)
                    peak.height.append(this_peak.height)
                    if this_peak.mu is not None:
                        peak.mu.append(this_peak.mu)
                    peak.sigma.append(this_peak.sigma)
                    peak.width.append(this_peak.width)

                    
    ## Calculate mean and error    
    for i, spectrum in enumerate(timescan.Y):
        
        timescan.BaselineSumError[i] = np.std(timescan.BaselineSum[i])/(np.size(timescan.BaselineSum[i]) ** 0.5)
        timescan.BaselineSum[i] = np.mean(timescan.BaselineSum[i])
        
        
        for peak in timescan.peaks[i]:
            
            peak.area_error = np.std(peak.area)/(np.size(peak.area) ** 0.5)
            peak.area = np.mean(peak.area)
            
            peak.baseline_error = np.std(peak.baseline)/(np.size(peak.baseline) ** 0.5)
            peak.baseline = np.mean(peak.baseline)
            
            peak.height_error = np.std(peak.height)/(np.size(peak.height) ** 0.5)
            peak.height = np.mean(peak.height)
            
            peak.mu_error = np.std(peak.mu)/(np.size(peak.mu) ** 0.5)
            peak.mu = np.mean(peak.mu)
            
            peak.sigma_error = np.std(peak.sigma)/(np.size(peak.sigma) ** 0.5)
            peak.sigma = np.mean(peak.sigma)
            
            peak.width_error = np.std(peak.width)/(np.size(peak.width) ** 0.5)
            peak.width = np.mean(peak.width)
            
#%% Decay fit of avg particles


for particle in avg_particles[3:4]:
    print('\n' + particle.name)
    timescan = particle.timescan
    peaks = timescan.peaks
    time = np.arange(0, len(timescan.Y) * timescan.chunk_size, timescan.chunk_size, dtype = int)   
    
    particle.ms = np.zeros(len(timescan.peaks[0]) + 1)
    particle.ts = np.zeros(len(timescan.peaks[0]) + 1)    
    particle.taus = np.zeros(len(timescan.peaks[0]) + 1)
    particle.bs = np.zeros(len(timescan.peaks[0]) + 1)
    
    for i, peak in enumerate(timescan.peaks[0]):
           
        ## Peak area ratios to plot for y
        y = [] 
        # ax = axes[np.where(peaks_list == i)[0][0]]    
        # ax.set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
                        
        for j in range(0, len(timescan.Y)):
            if peaks[j][i].error:
                y.append(nan)
            elif peaks[j][i].name == '1435/1420':
                y.append(peaks[j][i].area)   
            else:
                y.append(peaks[j][i].area/timescan.peaks[0][i].area)          
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = time, y = y) 
        
        ## Peak area decay
        spread = peak_spec.y[0] - peak_spec.y[-1]
        p0 = [spread * 0.9, 0.005, peak_spec.y[-1]]  
        # p0 = [peak_spec.y[0], 0.005, peak_spec.y[-1]]
        try:
            popt, pcov = curve_fit(f = exp_decay, 
                                    xdata = peak_spec.x,
                                    ydata = peak_spec.y,
                                    p0 = p0,
                                    bounds=(
                                        (spread * 0.85, 0.0001, 0),
                                        (spread * 1, 0.1, np.inf)),
                                    nan_policy = 'omit')
        except:
            popt = p0
            print('Peak Error: ' + str(peak.name))
            
        particle.ms[i] = popt[0]
        particle.ts[i] = popt[1]
        particle.taus[i] = 1/popt[1]
        particle.bs[i] = popt[2]
        fit_x = peak_spec.x
        fit_y = exp_decay(fit_x, *popt) 
        # plt.plot(fit_x, fit_y)                   
        # plt.plot(fit_x, peak_spec.y, alpha = 0.4)
        # plt.text(x = 500, y = fit_y.max() * 1.1, s = str(particle.taus[i]))
        # plt.show()
        
    ## Plot background sum
    y = []
    for j in range(0, len(timescan.Y)):
        y.append(timescan.BaselineSum[j])
    y = np.array(y)
    background_spec = SERS.SERS_Spectrum(x = time, y = y) 
    # background_spec.y = spt.butter_lowpass_filt_filt(background_spec.y, cutoff = 3000, fs = 40000)
    
    ## Background sum decay
    spread = background_spec.y[0] - background_spec.y[-1]
    p0 = [spread * 0.9, 0.005, background_spec.y[-1]]    
    try:
        popt, pcov = curve_fit(f = exp_decay, 
                            xdata = background_spec.x,
                            ydata = background_spec.y,
                            p0 = p0,
                            bounds=(
                                (spread * 0.85, 0.0001, 0),
                                (spread * 1, 0.1, np.inf)),
                            nan_policy = 'omit')
    except:
        popt = p0
        print('Background Error')
        
    particle.ms[-1] = popt[0]
    particle.ts[-1] = popt[1]
    particle.taus[-1] = 1/popt[1]
    particle.bs[-1] = popt[2]
    fit_x = background_spec.x
    fit_y = exp_decay(fit_x, *popt)       
    plt.plot(fit_x, fit_y)                   
    plt.plot(fit_x, background_spec.y)
    plt.text(x = 500, y = fit_y.max() * 1.1, s = str(particle.taus[-1]))
    plt.show()
    
    # print(particle.ms[-1]/(background_spec.y[0] - background_spec.y[-1]))
    # print(particle.bs[-1]/background_spec.y[-1])
    # plot_peak_areas(particle, save = False)
            
#%% Loop over avg particles and plot

for particle in tqdm(avg_particles, leave = True):
    
    plot_timescan(particle, save = True)
    plot_peak_areas(particle, save = True)
    plot_peak_positions(particle, save = True)
    plot_timescan_background(particle, save = True)
    
    
#%% Plot peak area tau v. laser power for average particles

fig, axes = plt.subplots(9,1,figsize=[14,22], sharex = True)
fig.suptitle('633 nm SERS - Peak Area Decay Time v. Laser Power', fontsize = 'x-large')
axes[len(axes)-1].set_xlabel('Laser Power ($\mu$W)', size = 'x-large')

   
timescan = avg_particles[0].timescan

colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'deeppink', 'black']

# fig, axes = plt.subplots(len(peaks_list),1,figsize=[12,30], sharex = True)
# fig.suptitle('633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W', fontsize = 'x-large')
# axes[len(axes)-1].set_xlabel('Dark time (s)', size = 'x-large')
peaks_list = np.array([0,1,2,3,5,6,7,8])
for i, peak in enumerate(timescan.peaks[0]):
    
    if i in peaks_list:
    
        ## Plot each peak recovery on own axis
        ax = axes[np.where(peaks_list == i)[0][0]] 
        ax.set_xticks(np.linspace(0, 26, 14))
        ax.set_ylabel('$\\tau$ I$_{SERS}$', size = 'x-large')
        ax.set_xlim(0, 26)
              
        ## Get tau v. laser power
        y = []
        for particle in avg_particles:
            y.append(particle.taus[i])            
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = np.array(laser_powers) * 1000, y = y)
        color = colors[np.where(peaks_list == i)[0][0]]
        ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 1)
        # ax.scatter(peak_spec.x, peak_spec.y, marker = 'o', facecolors = 'none', edgecolors = color, linewidth = 3, s = 150, zorder = 2)

        ax.legend(loc = 'upper right')
    
# ax.set_xlim(-100, 1000)
# ax.set_xscale('symlog')
y = []
for particle in avg_particles:
    if '3.0' not in particle.name:
        y.append(particle.taus[-1])     
    else:
        y.append(nan)
           
y = np.array(y)
peak_spec = SERS.SERS_Spectrum(x = np.array(laser_powers) * 1000, y = y)
color = 'black'
axes[-1].plot(peak_spec.x, peak_spec.y, label = 'Background', color = color, zorder = 1)
axes[-1].set_ylabel('$\\tau_{Background}$', size = 'x-large')
axes[-1].legend(loc = 'upper right')

plt.tight_layout(pad = 1.5)
    
## Save
save = True
if save == True:
    save_dir = get_directory('Compiled')
    fig.savefig(save_dir + 'Tau Peak Area' + '.svg', format = 'svg')
    plt.close(fig)


#%% Plot peak area equilibrium (b) v. laser power for average particles

fig, axes = plt.subplots(9,1,figsize=[14,22], sharex = True)
fig.suptitle('633 nm SERS - Peak Area Equilibrium v. Laser Power', fontsize = 'x-large')
axes[len(axes)-1].set_xlabel('Laser Power ($\mu$W)', size = 'x-large')

   
timescan = avg_particles[0].timescan

colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'deeppink', 'black']

# fig, axes = plt.subplots(len(peaks_list),1,figsize=[12,30], sharex = True)
# fig.suptitle('633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W', fontsize = 'x-large')
# axes[len(axes)-1].set_xlabel('Dark time (s)', size = 'x-large')
peaks_list = np.array([0,1,2,3,5,6,7,8])
for i, peak in enumerate(timescan.peaks[0]):
    
    if i in peaks_list:
    
        ## Plot each peak recovery on own axis
        ax = axes[np.where(peaks_list == i)[0][0]] 
        ax.set_xticks(np.linspace(0, 26, 14))
        ax.set_ylabel('Eq. I$_{SERS}$', size = 'x-large')
        ax.set_xlim(0, 26)
              
        ## Get tau v. laser power
        y = []
        for particle in avg_particles:
            y.append(particle.bs[i])            
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = np.array(laser_powers) * 1000, y = y)
        color = colors[np.where(peaks_list == i)[0][0]]
        ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 1)
        # ax.scatter(peak_spec.x, peak_spec.y, marker = 'o', facecolors = 'none', edgecolors = color, linewidth = 3, s = 150, zorder = 2)

        ax.legend(loc = 'upper right')
    
# ax.set_xlim(-100, 1000)
# ax.set_xscale('symlog')
y = []
for particle in avg_particles:
    if '3.0' not in particle.name:
        y.append(particle.bs[-1])            
    else:
        y.append(nan)
y = np.array(y)
peak_spec = SERS.SERS_Spectrum(x = np.array(laser_powers) * 1000, y = y)
color = 'black'
axes[-1].plot(peak_spec.x, peak_spec.y, label = 'Background', color = color, zorder = 1)
axes[-1].set_ylabel('Eq.$_{Background}$', size = 'x-large')
axes[-1].legend(loc = 'upper right')

plt.tight_layout(pad = 1.5)
    
## Save
save = True
if save == True:
    save_dir = get_directory('Compiled')
    fig.savefig(save_dir + 'Equilibrium Peak Area' + '.svg', format = 'svg')
    plt.close(fig)
