# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:23:15 2024

@author: il322


Plotter for Co-TAPP-SMe 633nm SERS Powerswitch & Recovery
(copied from 2024-04-10_633nm_Powerswitch_Recovery_Plotter.py)
(Minor adjustments to accommodate different dataset)
- Single, Double, & Triple Gaussian peak fitting
- Stores data in Particle() class
- Plot timescans with dark time, backgroun timescans, fitted peak area & position of several peaks, 
  and recovery of peak area and background with switching  


Data: 2024-04-10_633nm-BPT-MLAgg-Powerseries_633nm-Co-TAPP-SMe_MLAgg-Powerswitch_CCD-Flatness.h5


(samples:
     2023-11-28_Co-TAPP-SMe_60nm_MLAgg_b)

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
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2023-12-19_633nm_SERS_400Grating_Powerswitch_VariedDarkTime.h5")


#%% Spectral calibration

# Spectral calibration

## Get default literature BPT spectrum & peaks
lit_spectrum, lit_wn = cal.process_default_lit_spectrum()

## Load BPT ref spectrum
bpt_ref = my_h5['ref_meas']['BPT_633nm']
bpt_ref = SERS.SERS_Spectrum(bpt_ref)

## Coarse adjustments to miscalibrated spectra
# coarse_shift = 100 # coarse shift to ref spectrum
# coarse_stretch = 0.91 # coarse stretch to ref spectrum
coarse_shift = 0
coarse_stretch = 1
notch_range = [(70 + coarse_shift) * coarse_stretch, (128 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
truncate_range = [notch_range[1] + 100, None] # Truncate range for all spectra on this calibration - Add 50 to take out notch slope

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
# ref_wn = cal.find_ref_peaks(bpt_ref_no_notch, lit_spectrum = lit_spectrum, lit_wn = lit_wn, threshold = 0.05, distance = 1)
ref_wn = [0,0,0,0,0,0,0]
ref_wn[0] = bpt_ref_no_notch.x[62]
ref_wn[1] = bpt_ref_no_notch.x[87]
ref_wn[2] = bpt_ref_no_notch.x[150]
ref_wn[3] = bpt_ref_no_notch.x[274]
ref_wn[4] = bpt_ref_no_notch.x[306]
ref_wn[5] = bpt_ref_no_notch.x[385]
ref_wn[6] = bpt_ref_no_notch.x[507]


## Find calibrated wavenumbers
wn_cal = cal.calibrate_spectrum(bpt_ref_no_notch, ref_wn, lit_spectrum = lit_spectrum, lit_wn = lit_wn, linewidth = 1, deg = 3)
bpt_ref.x = wn_cal


#%% Spectral efficiency white light calibration

white_ref = my_h5['ref_meas']['white_ref']
white_ref = SERS.SERS_Spectrum(white_ref.attrs['wavelengths'], white_ref[2], title = 'White Scatterer')

## Convert to wn
white_ref.x = spt.wl_to_wn(white_ref.x, 632.8)
white_ref.x = white_ref.x + coarse_shift
white_ref.x = white_ref.x * coarse_stretch

## Get white bkg (counts in notch region)
# notch = SERS.SERS_Spectrum(white_ref.x[np.where(white_ref.x < (notch_range[1]-50))], white_ref.y[np.where(white_ref.x < (notch_range[1] - 50))], name = 'White Scatterer Notch') 
# notch = SERS.SERS_Spectrum(x = spt.truncate_spectrum(white_ref.x, white_ref.y, notch_range[0], notch_range[1] - 100)[0], 
#                             y = spt.truncate_spectrum(white_ref.x, white_ref.y, notch_range[0], notch_range[1] - 100)[1], 
#                             name = 'White Scatterer Notch')
# notch = SERS.SERS_Spectrum(white_ref.x, white_ref.y, title = 'Notch')
# notch_range = [(70 + coarse_shift) * coarse_stretch, (128 + coarse_shift) * coarse_stretch] # Define notch range as region in wavenumbers
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
                                    white_bkg = 420,
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
        
    directory_path = r'C:\Users\il322\Desktop\Offline Data\2023-12-19_633nm Powerswitch Recovery Analysis NEW\\' + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%% Testing background subtraction & cosmic ray removal

dark_counts = 420

particle = my_h5['ParticleScannerScan_4']['Particle_20']


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
    # spectrum.x = spectrum.x - 1.5
    spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
    spectrum.x = spectrum.x + coarse_shift
    spectrum.x = spectrum.x * coarse_stretch
    spectrum.truncate(start_x = truncate_range[0], end_x = truncate_range[1])
    spectrum.x = wn_cal
    spectrum.calibrate_intensity(R_setup = R_setup,
                                  dark_counts = dark_counts,#dark_powerseries[i].y,
                                  exposure = spectrum.cycle_time)
    
    spectrum.y_cosmic = spt.remove_cosmic_rays(spectrum.y, threshold = 9)

    ## Baseline
    spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
    spectrum.y_baselined = spectrum.y - spectrum.baseline
    spectrum.y_cosmic = spectrum.y_cosmic - spectrum.baseline
    
    ## Plot raw, baseline, baseline subtracted
    offset = 50000
    spectrum.plot(ax = ax, plot_y = (spectrum.y - spectrum.y.min()) + (i*offset), linewidth = 1, color = 'black', label = 'Raw', zorder = 1)
    spectrum.plot(ax = ax, plot_y = spectrum.y_baselined + (i*offset), linewidth = 1, color = 'blue', alpha = 1, label = 'Background subtracted', zorder = 2)
    spectrum.plot(ax = ax, plot_y = spectrum.baseline- spectrum.y.min() + (i*offset), title = 'Background Subtraction & Cosmic Ray Test', color = 'darkred', label = 'Background', linewidth = 1)    
    spectrum.plot(ax = ax, plot_y = spectrum.y_cosmic + (i*offset), title = 'Background Subtraction & Cosmic Ray Test', color = 'orange', label = 'Cosmic ray removed', linewidth = 1, linestyle = '--', zorder = 3)
    fig.suptitle(particle.name)    

# ax.set_xlim(450, 550)
ax.legend()
# ax.ticklabel_format(style = 'sci', scilimits = (0,0))
# ax.set_xlim(1200, 1700)
# ax.set_ylim(0, powerseries[].y_baselined.max() * 1.5)
plt.tight_layout(pad = 0.8)

#%% Testing calibration against 2024-04-10

# spectrum12 = SERS.SERS_Spectrum(spectrum.x, spectrum.y)
# spectrum12.truncate(400, 1750)
# spectrum12.normalise()
# bpt12 = SERS.SERS_Spectrum(bpt_ref.x, bpt_ref.y)
# bpt12.normalise()

#%% Stretch and shift CO-TAPP spectrum to match data from 2024-04-10...

stretch = 0.9913
shift = 26.5

# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('SERS Intensity (cts/mW/s)')
# ax.plot((spectrum12.x * stretch) + shift, spectrum12.y_norm, color = 'blue', label = 'Co-TAPP 12-19')
# ax.plot(spectrum04.x, spectrum04.y_norm, color = 'red', label = 'Co-TAPP 04-10')
# # ax.plot(bpt12.x, bpt12.y_norm+1, color = 'darkblue', label = 'BPT 12-19')
# # ax.plot(bpt04.x, bpt04.y_norm+1, color = 'darkred', label = 'BPT 04-10', alpha = 0.3)
# ax.legend()
# ax.set_xlim(400, 550)


# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('SERS Intensity (cts/mW/s)')
# ax.plot((spectrum12.x * stretch) + shift, spectrum12.y_norm, color = 'blue', label = 'Co-TAPP 12-19')
# ax.plot(spectrum04.x, spectrum04.y_norm, color = 'red', label = 'Co-TAPP 04-10')
# # ax.plot(bpt12.x, bpt12.y_norm+1, color = 'darkblue', label = 'BPT 12-19')
# # ax.plot(bpt04.x, bpt04.y_norm+1, color = 'darkred', label = 'BPT 04-10', alpha = 0.3)
# ax.legend()
# ax.set_xlim(1610, 1640)


# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('SERS Intensity (cts/mW/s)')
# ax.plot((spectrum12.x * stretch) + shift, spectrum12.y_norm, color = 'blue', label = 'Co-TAPP 12-19')
# ax.plot(spectrum04.x, spectrum04.y_norm, color = 'red', label = 'Co-TAPP 04-10')
# # ax.plot(bpt12.x, bpt12.y_norm+1, color = 'darkblue', label = 'BPT 12-19')
# # ax.plot(bpt04.x, bpt04.y_norm+1, color = 'darkred', label = 'BPT 04-10', alpha = 0.3)
# ax.legend()
# ax.set_xlim(1200, 1450)


# fig, ax = plt.subplots(1,1,figsize=[12,9])
# ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# ax.set_ylabel('SERS Intensity (cts/mW/s)')
# ax.plot((spectrum12.x * stretch) + shift, spectrum12.y_norm, color = 'blue', label = 'Co-TAPP 12-19')
# ax.plot(spectrum04.x, spectrum04.y_norm, color = 'red', label = 'Co-TAPP 04-10')
# fig.suptitle('Stretch = 0.9913, shift = 26.5')
# # ax.plot(bpt12.x, bpt12.y_norm+1, color = 'darkblue', label = 'BPT 12-19')
# # ax.plot(bpt04.x, bpt04.y_norm+1, color = 'darkred', label = 'BPT 04-10', alpha = 0.3)
# ax.legend()
# # ax.set_xlim(1200, 1450)

# # fig, ax = plt.subplots(1,1,figsize=[12,9])
# # ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
# # ax.set_ylabel('SERS Intensity (cts/mW/s)')
# # ax.plot((spectrum12.x * 0.995) + 20, spectrum12.y_norm, color = 'blue', label = 'Co-TAPP 12-19')
# # ax.plot(spectrum04.x, spectrum04.y_norm, color = 'red', label = 'Co-TAPP 04-10')
# # # ax.plot(bpt12.x, bpt12.y_norm+1, color = 'darkblue', label = 'BPT 12-19')
# # # ax.plot(bpt04.x, bpt04.y_norm+1, color = 'darkred', label = 'BPT 04-10', alpha = 0.3)
# # ax.legend()
# # ax.set_xlim(400, 550)

#%% Get all particles to analyze into Particle class with h5 locations and in a list


particles = []

scan_list = ['ParticleScannerScan_4']

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
                                      dark_counts = dark_counts,#dark_powerseries[i].y,
                                      exposure = spectrum.cycle_time,
                                      laser_power = spectrum.laser_power)
        
        spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
        spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
        spectrum.y_baselined = spectrum.y - spectrum.baseline
        spectrum.normalise(norm_y = spectrum.y_baselined)
        powerseries[i] = spectrum

    return powerseries


def process_powerswitch_recovery(particle):
    
    particle.dark_time = float(np.array(particle.h5_address['dark_time']))
    
    ## Add all SERS spectra to powerseries list in order
    keys = list(particle.h5_address.keys())
    keys = natsort.natsorted(keys)
    powerseries = []
    for key in keys:
        if 'SERS_Before' in key:
            powerseries.append(particle.h5_address[key])
    for key in keys:
        if 'SERS_After' in key:
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
        spectrum.x = (spectrum.x * stretch) + shift ## Extra shifting to match calibration from 2024-04-10 data
        spectrum.calibrate_intensity(R_setup = R_setup,
                                      dark_counts = dark_counts,#dark_powerseries[i].y,
                                      exposure = spectrum.cycle_time,
                                      laser_power = spectrum.laser_power)
        # spectrum.truncate(None, 1800)
        spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
        spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
        spectrum.baseline_sum = np.sum(spectrum.baseline)
        spectrum.y_baselined = spectrum.y - spectrum.baseline
        # spectrum.normalise(norm_y = spectrum.y_baselined)
        powerseries[i] = spectrum
    
    particle.powerseries = powerseries
        


#%% Loop over all particles and process powerswitch

print('\nProcessing spectra...')
for particle in tqdm(particles, leave = True):
    
    process_powerswitch_recovery(particle)


#%% Peak fitting functions
       
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
        

def fit_gaussian_powerseries(particle, fit_range, peak_name = None, R_sq_thresh = 0.85, smooth_first = False, plot = False):

    '''
    Fit single gaussian peak to given range of each spectrum in powerseries
    Adds peaks as attributes to each spectra in powerseries
    '''    
    
    powerseries = particle.powerseries

    for i, spectrum in enumerate(powerseries):
        
    
        ## Get region of spectrum to fit
        fit_range_index = [np.abs(spectrum.x-fit_range[0]).argmin(), np.abs(spectrum.x-fit_range[1]).argmin()+1]
        x = spectrum.x[fit_range_index[0]:fit_range_index[1]]
        y = spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]]
       
        if smooth_first == True:
            spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 3000, fs = 30000)
            y = spectrum.y_smooth[fit_range_index[0]:fit_range_index[1]]
        
    
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
        
        ## Add peak class to spectrum class 'peaks' list
        particle.powerseries[i].peaks.append(this_peak)
           
        
        # Plot
        
        if plot == True: # or error == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title(spectrum.name, fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
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

# particle = particles[1]
# fit_gaussian_powerseries(particle = particle, fit_range = [480, 545], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1100, 1150], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1170, 1220], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1280, 1310], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1475, 1525], smooth_first = True, plot = False)
# fit_gaussian_powerseries(particle = particle, fit_range = [1628, 1655], smooth_first = True,  plot = True)    
# fit_gaussian_powerseries(particle = particle, fit_range = [1425, 1450], smooth_first = True,  plot = True)    

 
#%% Peak fit for double Gaussian region

# def fit_gaussian2_powerseries(particle, fit_range = [1410, 1450], peak_name = None, R_sq_thresh = 0.85, smooth_first = False, plot = False):

#     '''
#     Fit single gaussian peak to given range of each spectrum in powerseries
#     Adds peaks as attributes to each spectra in powerseries
#     '''    
    
#     powerseries = particle.powerseries

#     for i, spectrum in enumerate(powerseries):
        
    
#         ## Get region of spectrum to fit
#         fit_range_index = [np.abs(spectrum.x-fit_range[0]).argmin(), np.abs(spectrum.x-fit_range[1]).argmin()+1]
#         x = spectrum.x[fit_range_index[0]:fit_range_index[1]]
#         y = spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]]
       
#         if smooth_first == True:
#             spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 5000, fs = 15000)
#             y = spectrum.y_smooth[fit_range_index[0]:fit_range_index[1]]
        
    
#         # Fit
        
#         # ## Initial guesses
#         # i_max = y.argmax() # index of highest value - for guess, assumed to be Gaussian peak
#         # height0 = y[i_max]
#         # mu0 = x[i_max] # centre x position of guessed peak
#         # baseline0 = (y[0] + y[len(y)-1])/2 # height of baseline guess
#         # i_half = np.argmax(y >= (height0 + baseline0)/2) # Index of first argument to be at least halfway up the estimated bell
#         # # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
#         # # half, and even if this point is a distant outlier the fit should still converge.
#         # sigma0 = (mu0 - x[i_half]) / np.sqrt(2*np.log((height0 - baseline0)/(y[i_half] - baseline0)))
#         # width0 = sigma0 * 2 * np.sqrt(2*np.log(2))
#         # # sigma0 = 3
#         # p0 = height0, mu0, width0, baseline0
        
#         ## Perform fit (height, mu, width, baseline)
#         start = time.time()
        
#         ## Initial guesses
        
#         ### First peak
#         height1 = y.max()/1.7
#         mu1 = 1420
#         width1 = 15
#         baseline1 = 0
        
#         ### Second peak
#         height2 = y.max()
#         mu2 = 1434
#         width2 = 17
#         baseline2 = 0
        
        
#         p0 = [
#                 height1, mu1, width1, baseline1,
#                 height2, mu2, width2, baseline2
#              ]
        
#         # lower_bounds = (
#         #                 0, mu1 - 5, width1 - 5, y.min(),
#         #                 0, mu2 - 1, width2 - 2, y.min(),
#         #                 0, mu3 - 1, width3 - 5, y.min()
#         #                )
        
#         # upper_bounds = (
#         #                 height1 * 2, mu1 + 5, width1 + 5, 1000,
#         #                 height2 * 2, mu2 + 1, width2 + 2, 1000,
#         #                 height3 * 2, mu3 + 1, width3 + 5, 1000
#         #                )

#         lower_bounds = (
#                         0, x.min(), 0, y.min(),
#                         0, x.min(), 0, y.min()
#                        )
        
#         upper_bounds = (
#                         height1 * 2, x.max(), x.max()-x.min(), 1000,
#                         height2 * 2, x.max(), x.max()-x.min(), 1000
#                        )        
        
#         try:
#             popt, pcov = curve_fit(f = gaussian2, 
#                                 xdata = x,
#                                 ydata = y, 
#                                 p0 = p0,
#                                 bounds=((lower_bounds),(upper_bounds),),)
            
#         except:
#             popt = p0
#             print('\nFit Error')
          
#         finish = time.time()
#         runtime = finish - start
            
#         # print('Guess:', np.array(p0))
#         # print('Fit:  ', popt)
        
        
#         # Get fit data
        
#         ## Peak 1
#         height = popt[0]
#         mu = popt[1]
#         width = popt[2]
#         baseline = popt[3]
#         sigma = width / (2 * np.sqrt(2 * np.log(2)))
#         area = height * sigma * np.sqrt(2 * np.pi)
#         y_fit = gaussian(x, *popt[0:4])
#         if peak_name is None:
#             peak_name = str(int(np.round(np.mean(fit_range),0)))
#         peak1 = Peak(*popt[0:4])
#         peak1.area = area
#         peak1.sigma = sigma
#         peak1.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
#         peak1.name = 'Triple A'

#         ## Peak 2
#         height = popt[4]
#         mu = popt[5]
#         width = popt[6]
#         baseline = popt[7]
#         sigma = width / (2 * np.sqrt(2 * np.log(2)))
#         area = height * sigma * np.sqrt(2 * np.pi)
#         y_fit = gaussian(x, *popt[4:8])
#         if peak_name is None:
#             peak_name = str(int(np.round(np.mean(fit_range),0)))
#         peak2 = Peak(*popt[4:8])
#         peak2.area = area
#         peak2.sigma = sigma
#         peak2.spectrum = SERS.SERS_Spectrum(x = x, y = y_fit)
#         peak2.name = 'Triple B'

        
#         ## Sum & residuals
#         y_fit = peak1.spectrum.y + peak2.spectrum.y
#         residuals = y - y_fit
#         residuals_sum = np.abs(residuals).sum()
#         R_sq =  1 - ((np.sum(residuals**2)) / (np.sum((y - np.mean(y))**2)))
#         ### R_sq is total R_sq for triple
#         peak1.R_sq = R_sq
#         peak2.R_sq = R_sq

#         # if peak_name is None:
#         #     peak_name = str(int(np.round(np.mean(fit_range),0)))
        
#         ## Screening poor fits
#         if R_sq < R_sq_thresh or np.isnan(R_sq):
#             print('\nPoor Fit')
#             print(particle.name)
#             print(spectrum.name)
#             print('R^2 = ' + str(np.round(R_sq, 3)))
#             print('Guess:', np.array(p0))
#             print('Fit:  ', popt)
#             error = True
#         else:
#             error = False
#         peak1.error = error
#         peak2.error = error    
        
#         ## Add peaks class to spectrum class 'peaks' list
#         particle.powerseries[i].peaks.append(peak1)
#         particle.powerseries[i].peaks.append(peak2)   
        
        
#         # Plot
        
#         if plot == True or error == True:
            
#             fig, ax = plt.subplots(1,1,figsize=[18,9])
#             fig.suptitle(particle.name, fontsize = 'large')
#             ax.set_title(spectrum.name, fontsize = 'large')
#             ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
#             ax.set_ylabel('SERS Intensity (cts/mW/s)')
#             if smooth_first:
#                 ax.plot(x, spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
#                 ax.plot(x, y, color = 'grey', label = 'Smoothed') 
#             else:
#                 ax.plot(x, y, color = 'grey', linestyle = '--', label = 'Data')
#             ax.plot(x, peak1.spectrum.y, label = 'Fit A', color = 'red')
#             ax.plot(x, peak2.spectrum.y, label = 'Fit B', color = 'green')
#             ax.plot(x, y_fit, label = 'Fit', color = 'orange')
#             ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
#             ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
#             ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
#             ax.set_xlim(fit_range[0], fit_range[1])
#             ax.set_ylim(None, y.max()*1.2)
#             ax.legend()
#             plt.show()


# # Testing fit ranges

# particle = particles[10]
# for spectrum in particle.powerseries:
#     spectrum.peaks = []
# fit_gaussian_powerseries(particle = particle, fit_range = [480, 545], smooth_first = True, plot = False)
# fit_gaussian_powerseries(particle = particle, fit_range = [1100, 1150], smooth_first = True, plot = False)
# fit_gaussian_powerseries(particle = particle, fit_range = [1170, 1220], smooth_first = True, plot = False)
# fit_gaussian_powerseries(particle = particle, fit_range = [1280, 1310], smooth_first = True, plot = False)
# fit_gaussian_powerseries(particle = particle, fit_range = [1475, 1525], smooth_first = True, plot = False)
# fit_gaussian_powerseries(particle = particle, fit_range = [1628, 1655], smooth_first = True,  plot = False)    
# fit_gaussian2_powerseries(particle = particle, fit_range = [1415, 1450], smooth_first = True, plot = True)
# plot_peak_areas(particle)   


#%% Peak fit for triple Gaussian region

def fit_gaussian3_powerseries(particle, fit_range = [1365, 1450], peak_name = None, R_sq_thresh = 0.85, smooth_first = False, plot = False, save = False):

    '''
    Fit single gaussian peak to given range of each spectrum in powerseries
    Adds peaks as attributes to each spectra in powerseries
    '''    
    
    powerseries = particle.powerseries

    for i, spectrum in enumerate(powerseries):
        
    
        ## Get region of spectrum to fit
        fit_range_index = [np.abs(spectrum.x-fit_range[0]).argmin(), np.abs(spectrum.x-fit_range[1]).argmin()+1]
        x = spectrum.x[fit_range_index[0]:fit_range_index[1]]
        y = spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]]
       
        if smooth_first == True:
            spectrum.y_smooth = spt.butter_lowpass_filt_filt(spectrum.y_baselined, cutoff = 5000, fs = 15000)
            y = spectrum.y_smooth[fit_range_index[0]:fit_range_index[1]]

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
        
        ## Add peaks class to spectrum class 'peaks' list
        particle.powerseries[i].peaks.append(peak1)
        particle.powerseries[i].peaks.append(peak2)   
        particle.powerseries[i].peaks.append(peak3)
        
        
        # Plot
        
        if plot == True: # or error == True:
            
            fig, ax = plt.subplots(1,1,figsize=[18,9])
            fig.suptitle(particle.name, fontsize = 'large')
            ax.set_title(spectrum.name, fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
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
            ax.set_title(spectrum.name, fontsize = 'large')
            ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
            ax.set_ylabel('SERS Intensity (cts/mW/s)')
            if smooth_first:
                ax.plot(x, spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]], color = 'grey', linestyle = '--', label = 'Data')
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
            fig.savefig(save_dir + particle.name + '_' + particle.powerseries[i].name + '_TripleFit' + '.svg', format = 'svg')
            plt.close(fig)


# Testing fit ranges

particle = particles[1]
# for spectrum in particle.powerseries:
#     spectrum.peaks = []
# fit_gaussian_powerseries(particle = particle, fit_range = [480, 545], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1100, 1150], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1170, 1220], smooth_first = True, plot = True)
# fit_gaussian_powerseries(particle = particle, fit_range = [1280, 1310], smooth_first = True, plot = True)
# fit_gaussian3_powerseries(particle = particle, fit_range = [1365, 1450], smooth_first = True, plot = True, save = False)
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
    
    ## Clear previous peaks    
    for spectrum in particle.powerseries:
        spectrum.peaks = []
        
    ## Fit peaks
    fit_gaussian_powerseries(particle = particle, fit_range = [480, 545], smooth_first = True, plot = False)
    fit_gaussian_powerseries(particle = particle, fit_range = [1100, 1150], smooth_first = True, plot = False)
    fit_gaussian_powerseries(particle = particle, fit_range = [1170, 1220], smooth_first = True, plot = False)
    fit_gaussian_powerseries(particle = particle, fit_range = [1280, 1310], smooth_first = True, plot = False)
    fit_gaussian3_powerseries(particle = particle, fit_range = [1365, 1450], smooth_first = True, plot = False, save = False)
    # fit_gaussian_powerseries(particle = particle, fit_range = [1475, 1525], smooth_first = True, plot = False)
    fit_gaussian_powerseries(particle = particle, fit_range = [1628, 1655], smooth_first = True,  plot = False)    
    
    ## 1435/1425 ratio peak
    for spectrum in particle.powerseries:
        try:
            baseline = spectrum.peaks[6].baseline/spectrum.peaks[5].baseline
        except:
            baseline = 0
        ratio = Peak(height = spectrum.peaks[6].height/spectrum.peaks[5].height,
                     mu = None,
                     width = spectrum.peaks[6].width/spectrum.peaks[5].width,
                     baseline = baseline)
        ratio.area = spectrum.peaks[6].area/spectrum.peaks[5].area
        ratio.sigma = spectrum.peaks[6].sigma/spectrum.peaks[5].sigma
        ratio.name = '1435/1420'
        if spectrum.peaks[6].error or spectrum.peaks[5].error:
            ratio.error = True
        else:
            ratio.error = False
        spectrum.peaks.append(ratio)


# Report number of failed fits

count = 0
for particle in particles:
    for spectrum in particle.powerseries:
        for peak in spectrum.peaks:
            if peak.error: count += 1
print('\nFit Errors (%): ')
print(100*count/ (len(particles) * len(particles[0].powerseries) * len(spectrum.peaks)))
  
#%% Plotting functions

    
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
        fig.savefig(save_dir + particle.name + '_785nm Min Powerseries' + '.svg', format = 'svg')
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
        fig2.savefig(save_dir + particle.name + '_785nm Direct Powerseries' + '.svg', format = 'svg')
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
        fig3.savefig(save_dir + particle.name + '_633nm Powerswitch Timescan' + '.svg', format = 'svg')
        plt.close(fig)
        
def plot_timescan_powerswitch_recovery(particle, save = False):
    
    
    powerseries = particle.powerseries
    
    ## Add dark times into powerseries (for plotting)
    if particle.dark_time > 0:
        spectrum = SERS.SERS_Spectrum(x = powerseries[0].x, y = np.zeros(len(powerseries[0].y)))
        spectrum.y_baselined = spectrum.y        
        powerseries = np.insert(powerseries, [10], spectrum)
        # powerseries = np.insert(powerseries, [10], spectrum)
        # powerseries = np.insert(powerseries, [10], spectrum)
        # powerseries = np.insert(powerseries, [23], spectrum)
        # powerseries = np.insert(powerseries, [23], spectrum)
        # powerseries = np.insert(powerseries, [23], spectrum)
        # powerseries = np.insert(powerseries, [36], spectrum)
        # powerseries = np.insert(powerseries, [36], spectrum)
        # powerseries = np.insert(powerseries, [36], spectrum)

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
        ax3.text(x=820,y=9.8,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
        # ax3.text(x=820,y=23.5,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
        # ax3.text(x=820,y=36.5,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
    ax3.set_title('633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W' + '\n' + str(particle.name), fontsize = 'x-large', pad = 10)
    pcm = ax3.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig3.colorbar(pcm, ax=ax3)
    clb.set_label(label = 'SERS Intensity', size = 'large', rotation = 270, labelpad=30)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig3.savefig(save_dir + particle.name + '_633nm Powerswitch' + '.svg', format = 'svg')
        plt.close(fig3)
        
        
def plot_peak_areas(particle, save = False):
    
    
    powerseries = particle.powerseries
    
    scan = np.arange(0, len(powerseries), 1, dtype = int)   
    
    
    peaks_list = np.array([0,1,2,3,5,6,7,8])
    
    colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'black']
    
    fig, axes = plt.subplots(len(peaks_list),1,figsize=[12,30], sharex = True)
    fig.suptitle('633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W' + '\n' + str(particle.name) + '\n' + 'Dark Time  = ' + str(np.round(particle.dark_time,2)) + 's', fontsize = 'x-large')
    axes[len(axes)-1].set_xlabel('Scan No.', size = 'x-large')
    
    # axes[int(len(axes)/2)].set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
    
    for i, peak in enumerate(powerseries[0].peaks):
        
        if i in peaks_list:

            ## Peak area ratios to plot for y
            y = [] 
            ax = axes[np.where(peaks_list == i)[0][0]]    
            ax.set_xticks(scan[::2])
            ax.set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
                        
            for spectrum in powerseries:
                if spectrum.peaks[i].error:
                    y.append(nan)
                elif spectrum.peaks[i].name == '1435/1420':
                    y.append(spectrum.peaks[i].area)
                else:
                    y.append(spectrum.peaks[i].area/powerseries[0].peaks[i].area)          
            y = np.array(y)
            peak_spec = SERS.SERS_Spectrum(x = scan, y = y)
                                  
            ## Plot
            color = colors[np.where(peaks_list == i)[0][0]]
            ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 4)
            ax.scatter(peak_spec.x[1::2], peak_spec.y[1::2], label = '90 $\mu$W', marker = 'o', facecolors = color, edgecolors = color, linewidth = 3, s = 150, zorder = 2)
            ax.scatter(peak_spec.x[::2], peak_spec.y[::2], label = '1 $\mu$W', marker = 'o', facecolors = 'white', edgecolors = color, linewidth = 3, s = 150, zorder = 2)
       
            ## Errorbars, if avg particle
            try:                
                y_error = []
                for spectrum in powerseries:
                    if spectrum.peaks[i].error:
                        y_error.append(nan)
                    elif spectrum.peaks[i].name == '1435/1420':
                        y_error.append(spectrum.peaks[i].area_error)
                    else:
                        this_error = (spectrum.peaks[i].area/powerseries[0].peaks[i].area) * np.sqrt((spectrum.peaks[i].area_error/powerseries[i].peaks[i].area)**2 + (powerseries[0].peaks[i].area_error/powerseries[0].peaks[i].area)**2)
                        
                        y_error.append(this_error)   
                y_error = np.array(y_error)
                ax.errorbar(peak_spec.x[1::2], peak_spec.y[1::2], yerr = y_error[1::2], marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
                ax.errorbar(peak_spec.x[::2], peak_spec.y[::2], yerr = y_error[::2], marker = 'none', mfc = 'white', mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
            except:
                y_error = None

            ax.legend(loc = 'upper right')
        
            ylim = ax.get_ylim()            
            if particle.dark_time > 0:
                ax.vlines(9.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
                # ax.vlines(19.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
                # ax.vlines(29.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
            ax.set_ylim(ylim)
    
    axes[len(axes)-1].set_ylabel('I$_{1435}$ / I$_{1420}$', size = 'x-large')
    plt.tight_layout(pad = 1.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_Peak Area' + '.svg', format = 'svg')
        plt.close(fig)
        
        
def plot_peak_positions(particle, save = False):
    powerseries = particle.powerseries
    
    scan = np.arange(0, len(powerseries), 1, dtype = int)   
    
    
    peaks_list = np.array([0,1,2,3,5,6,7])
    
    colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'black']
    
    fig, axes = plt.subplots(len(peaks_list),1,figsize=[12,30], sharex = True)
    fig.suptitle('633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W' + '\n' + str(particle.name) + '\n' + 'Dark Time  = ' + str(np.round(particle.dark_time,2)) + 's', fontsize = 'x-large')
    axes[len(axes)-1].set_xlabel('Scan No.', size = 'x-large')
    
    # axes[int(len(axes)/2)].set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
    
    for i, peak in enumerate(powerseries[0].peaks):
        
        if i in peaks_list:

            y = []
            
            ax = axes[np.where(peaks_list == i)[0][0]]    
            ax.set_xticks(scan[::2])
            ax.set_ylabel('Peak Position (cm$^{-1}$)', size = 'x-large')
                        
            for spectrum in powerseries:
                if spectrum.peaks[i].error:
                    y.append(nan)
                else:
                    y.append(spectrum.peaks[i].mu)            
            y = np.array(y)
            peak_spec = SERS.SERS_Spectrum(x = scan, y = y)
            
            ## Plot
            color = colors[np.where(peaks_list == i)[0][0]]
            ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 1)
            ax.scatter(peak_spec.x[1::2], peak_spec.y[1::2], label = '90 $\mu$W', marker = 'o', facecolors = color, edgecolors = color, linewidth = 3, s = 150, zorder = 2)
            ax.scatter(peak_spec.x[::2], peak_spec.y[::2], label = '1 $\mu$W', marker = 'o', facecolors = 'none', edgecolors = color, linewidth = 3, s = 150, zorder = 2)
    
            ## Errorbars, if avg particle
            try:                
                y_error = []
                for spectrum in powerseries:
                    if spectrum.peaks[i].error:
                        y_error.append(nan)
                    else:
                        this_error = spectrum.peaks[i].mu_error
                        y_error.append(this_error)   
                        
                y_error = np.array(y_error)
                ax.errorbar(peak_spec.x[1::2], peak_spec.y[1::2], yerr = y_error[1::2], marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
                ax.errorbar(peak_spec.x[::2], peak_spec.y[::2], yerr = y_error[::2], marker = 'none', mfc = 'white', mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)
            except:
                y_error = None
            
            ax.legend(loc = 'upper right')
        
            ylim = ax.get_ylim()            
            if particle.dark_time > 0:
                ax.vlines(9.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
                # ax.vlines(19.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
                # ax.vlines(29.5, ylim[0], ylim[1], color = 'black', alpha = 0.5, linewidth = 10)
            ax.set_ylim(ylim)
            ax.ticklabel_format(useOffset = False, style = 'plain')

    plt.tight_layout(pad = 1.5)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_Peak Position' + '.svg', format = 'svg')
        plt.close(fig)
   
    
def plot_baseline_powerswitch_recovery_kinetic(particle, save = False):
    
    
    powerseries = particle.powerseries
    
    ## Add dark times into powerseries (for plotting)
    if particle.dark_time > 0:
        spectrum = SERS.SERS_Spectrum(x = powerseries[0].x, y = np.zeros(len(powerseries[0].y)))
        spectrum.y_baselined = spectrum.y  
        spectrum.baseline = np.zeros(spectrum.y.shape)
        powerseries = np.insert(powerseries, [10], spectrum)
        # powerseries = np.insert(powerseries, [10], spectrum)
        # powerseries = np.insert(powerseries, [10], spectrum)
        # powerseries = np.insert(powerseries, [23], spectrum)
        # powerseries = np.insert(powerseries, [23], spectrum)
        # powerseries = np.insert(powerseries, [23], spectrum)
        # powerseries = np.insert(powerseries, [36], spectrum)
        # powerseries = np.insert(powerseries, [36], spectrum)
        # powerseries = np.insert(powerseries, [36], spectrum)

    ## Get all specrta into single array for timescan
    powerseries_y = np.zeros((len(powerseries), len(powerseries[0].y)))
    for i,spectrum in enumerate(powerseries):
        powerseries_y[i] = spectrum.baseline
    powerseries_y = np.array(powerseries_y)

    ## Plot powerseries as timescan
    timescan = SERS.SERS_Timescan(x = spectrum.x, y = powerseries_y, exposure = 1)
    fig, (ax) = plt.subplots(1, 1, figsize=[16,16])
    t_plot = np.arange(0,len(powerseries),1)
    v_min = np.percentile(powerseries_y, 20)
    v_max = np.percentile(powerseries_y, 99.9)
    cmap = plt.get_cmap('inferno')
    ax.set_yticklabels([])
    ax.set_xlabel('Raman Shifts (cm$^{-1}$)', fontsize = 'large')
    # ax.set_xlim(450,2100)
    if particle.dark_time > 0:
        ax.text(x=820,y=9.75,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
        # ax.text(x=820,y=23.5,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
        # ax.text(x=820,y=36.5,s='Dark recovery time: ' + str(np.round(particle.dark_time,2)) + 's', color = 'white', size='x-large')
    ax.set_title('Background 633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W' + '\n' + str(particle.name), fontsize = 'x-large', pad = 10)
    pcm = ax.pcolormesh(timescan.x, t_plot, powerseries_y, vmin = v_min, vmax = v_max, cmap = cmap, rasterized = 'True')
    clb = fig.colorbar(pcm, ax = ax)
    clb.set_label(label = 'Intensity', size = 'large', rotation = 270, labelpad=30)
    
    ## Save
    if save == True:
        save_dir = get_directory(particle.name)
        fig.savefig(save_dir + particle.name + '_Background 633nm Powerswitch' + '.svg', format = 'svg')
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


particle = particles[8]
# plot_timescan_powerswitch_recovery(particle, save = False)
# plot_baseline_powerswitch_recovery_kinetic(particle, save = False)
# plot_peak_areas(particle, save = False)
plot_peak_positions(particle, save = False)
# plot_baseline_sum(particle, save = False)
        
# particle = avg_particles[0]
# plot_timescan_powerswitch_recovery(particle, save = False)
# plot_baseline_powerswitch_recovery_kinetic(particle, save = False)
# plot_peak_areas(particle, save = False)
# plot_peak_positions(particle, save = False)
# plot_baseline_sum(particle, save = False)


#%% Loop over all particles and plot

# print('\nPlotting...')
# for particle in tqdm(particles, leave = True):
    
#     plot_timescan_powerswitch_recovery(particle, save = True)     
#     plot_peak_areas(particle, save = True)
#     plot_peak_positions(particle, save = True)
#     plot_baseline_powerswitch_recovery_kinetic(particle, save = True)
#     plot_baseline_sum(particle, save = True)
    

#%% Making average particles & calculating powerseries

from copy import deepcopy

avg_particles = []
dark_times = []

## Get dark times
for particle in particles:
    if particle.dark_time not in dark_times:
        dark_times.append(particle.dark_time)


# Loop over each dark time - one avg particle per dark time
for dark_time in dark_times:
    ## Set up avg particle
    avg_particle = Particle()
    avg_particle.dark_time = dark_time
    avg_particle.name = 'MLAgg_Avg_' + str(int(avg_particle.dark_time)) + 's'
    avg_particle.powerseries = []
    
    ## Set up avg powerseries
    for i in range(0, len(particles[0].powerseries)):
        avg_particle.powerseries.append(SERS.SERS_Spectrum(x = particles[0].powerseries[0].x, y = np.zeros(particles[0].powerseries[0].y.shape)))
        avg_particle.powerseries[i].y_baselined = avg_particle.powerseries[i].y
        avg_particle.powerseries[i].baseline = deepcopy(avg_particle.powerseries[i].y)
    
    ## Add y-values to avg powerseries
    counter = 0
    for particle in particles:
        if particle.dark_time == avg_particle.dark_time:
            counter += 1
            for i in range(0, len(avg_particle.powerseries)):
                avg_particle.powerseries[i].y_baselined += particle.powerseries[i].y_baselined
                avg_particle.powerseries[i].baseline += particle.powerseries[i].baseline
    
    ## Divide
    for spectrum in avg_particle.powerseries:
        spectrum.y_baselined = spectrum.y_baselined/counter
        spectrum.baseline = spectrum.baseline/counter
                    
    avg_particles.append(avg_particle)
 
                      
#%% Get peak data for average particles


from copy import deepcopy

for avg_particle in avg_particles:
    
    
    ## Set up peaks list
    for spectrum in avg_particle.powerseries:
        
        spectrum.baseline_sum = []
        spectrum.baseline_sum_recovery = []
        
        spectrum.peaks = deepcopy(particles[0].powerseries[0].peaks)
        
        for peak in spectrum.peaks:
            peak.area = []
            peak.baseline = []
            peak.error = False
            peak.height = []
            peak.mu = []
            peak.sigma = []
            peak.width = []
            peak.recovery = []
    
    ## Add peaks
    counter = 0
    for particle in particles:
        if avg_particle.dark_time == particle.dark_time:
            counter += 1
            for i, spectrum in enumerate(avg_particle.powerseries):
                
                spectrum.baseline_sum.append(particle.powerseries[i].baseline_sum)
                spectrum.baseline_sum_recovery.append((particle.powerseries[10].baseline_sum - particle.powerseries[8].baseline_sum) / particle.powerseries[8].baseline_sum)
                
                for j, peak in enumerate(spectrum.peaks):
                    this_peak = deepcopy(particle.powerseries[i].peaks[j])
                    peak.area.append(this_peak.area)
                    peak.baseline.append(this_peak.baseline)
                    peak.height.append(this_peak.height)
                    if this_peak.mu is not None:
                        peak.mu.append(this_peak.mu)
                    peak.sigma.append(this_peak.sigma)
                    peak.width.append(this_peak.width)
                    peak.recovery.append((particle.powerseries[10].peaks[j].area - particle.powerseries[8].peaks[j].area) / particle.powerseries[8].peaks[j].area) # peak.recovery is saved to each spectrum in powerseries, but only has the value of spectrum 10 - spectrum 8 (before/after 1st dark recovery)
                    
    ## Calculate mean and error    
    for spectrum in avg_particle.powerseries:
        
        spectrum.baseline_sum_error = np.std(spectrum.baseline_sum)/(np.size(spectrum.baseline_sum) ** 0.5)
        spectrum.baseline_sum = np.mean(spectrum.baseline_sum)
        
        spectrum.baseline_sum_recovery_error = np.std(spectrum.baseline_sum_recovery)/(np.size(spectrum.baseline_sum_recovery) ** 0.5)
        spectrum.baseline_sum_recovery = np.mean(spectrum.baseline_sum_recovery)
        
        for peak in spectrum.peaks:
            
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
            
            peak.recovery_error = np.std(peak.recovery)/(np.size(peak.recovery) ** 0.5)
            peak.recovery = np.mean(peak.recovery)
            
            
#%% Loop over avg particles and plot

# for particle in avg_particles:
    
#     plot_timescan_powerswitch_recovery(particle, save = True)
#     plot_peak_areas(particle, save = True)
#     plot_peak_positions(particle, save = True)
#     plot_baseline_powerswitch_recovery_kinetic(particle, save = True)
#     plot_baseline_sum(particle, save = True)


#%% Plot peak area recovery v. dark_time (just for avg particles)
    
powerseries = avg_particles[0].powerseries

scan = np.arange(0, len(powerseries), 1, dtype = int)   

peaks_list = np.array([0,1,2,3,5,6,7,8])

colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'black']

fig, axes = plt.subplots(len(peaks_list),1,figsize=[12,30], sharex = True)
fig.suptitle('633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W', fontsize = 'x-large')
axes[len(axes)-1].set_xlabel('Dark time (s)', size = 'x-large')

for i, peak in enumerate(powerseries[0].peaks):
    
    if i in peaks_list:

        
        ## Plot each peak recovery on own axis
        ax = axes[np.where(peaks_list == i)[0][0]]    
        ax.set_xticks(np.linspace(dark_times[0], dark_times[-1], 11))
        ax.set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
              
        ## Get peak recovery v. scan
        y = []
        for particle in avg_particles:
                y.append(particle.powerseries[0].peaks[i].recovery)            
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = np.array(dark_times), y = y)
        color = colors[np.where(peaks_list == i)[0][0]]
        ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 1)
        ax.scatter(peak_spec.x, peak_spec.y, marker = 'o', facecolors = 'none', edgecolors = color, linewidth = 3, s = 150, zorder = 2)

        ## Errorbars           
        y_error = []
        for particle in avg_particles:
            this_error = (particle.powerseries[0].peaks[i].recovery_error)
                
            y_error.append(this_error)   
        y_error = np.array(y_error)
        ax.errorbar(peak_spec.x, peak_spec.y, yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)

        ax.legend(loc = 'upper right')
    
# ax.set_xlim(-100, 1000)
ax.set_xscale('symlog')
axes[len(axes)-1].set_ylabel('I$_{1435}$ / I$_{1420}$', size = 'x-large')
plt.tight_layout(pad = 1.5)
    
## Save
save = False
if save == True:
    save_dir = get_directory('Compiled')
    fig.savefig(save_dir + 'Peak Area Recovery' + '.svg', format = 'svg')
    plt.close(fig)
    
    
#%% Plot baseline sum recovery v. dark_time (just for avg particles)
    
powerseries = avg_particles[0].powerseries

scan = np.arange(0, len(powerseries), 1, dtype = int)   

fig, ax = plt.subplots(1, 1, figsize=[18,8], sharex = True)
fig.suptitle('633 nm Powerswitch Background Recovery - 1 $\mu$W / 90 $\mu$W', fontsize = 'x-large')
ax.set_xlabel('Dark time (s)', size = 'large')       
ax.set_xticks(np.linspace(dark_times[0], dark_times[-1], 11))
ax.set_ylabel('I$_{Background}$ (%$\Delta$)', size = 'x-large')
      
## Get background sum recovery v. scan
y = []
for particle in avg_particles:
        y.append(particle.powerseries[0].baseline_sum_recovery)            
y = np.array(y)
peak_spec = SERS.SERS_Spectrum(x = np.array(dark_times), y = y)
color = 'black'
ax.plot(peak_spec.x, peak_spec.y, color = color, zorder = 1)
ax.scatter(peak_spec.x, peak_spec.y, marker = 'o', facecolors = 'none', edgecolors = color, linewidth = 3, s = 150, zorder = 2)

## Errorbars           
y_error = []
for particle in avg_particles:
    this_error = (particle.powerseries[0].baseline_sum_recovery_error)
        
    y_error.append(this_error)   
y_error = np.array(y_error)
ax.errorbar(peak_spec.x, peak_spec.y, yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)

ax.legend(loc = 'upper right')
ax.set_xscale('symlog')
plt.tight_layout(pad = 0.5)
    
## Save
save = False
if save == True:
    save_dir = get_directory('Compiled')
    fig.savefig(save_dir + 'Background Sum Recovery' + '.svg', format = 'svg')
    plt.close(fig)
    

    
#%% Long spectrum plot to save

fig, ax = plt.subplots(1,1,figsize=[10,30], sharex = True)

spectrum = avg_particles[0].powerseries[0]
ax.plot(spectrum.y_baselined/spectrum.y_baselined.max(), spectrum.x, color = 'black', linewidth = 10)
ax.invert_yaxis()
ax.set_ylabel('Raman Shifts (cm$^{-1}$)', fontsize = 35, rotation = 270, labelpad = 75)
ax.yaxis.set_label_position("right")
ax.yaxis.set_ticks_position('right')
ax.tick_params(axis='y', labelsize=25)
ax.set_ylim(1680, 450)
ax.set_xticklabels([])

ax.hlines(512, -0.1, 1.1, color = 'grey', alpha = 0.5, linewidth = 70)
ax.hlines(1125, -0.1, 1.1, color = 'purple', alpha = 0.5, linewidth = 70)
ax.hlines(1195, -0.1, 1.1, color = 'brown', alpha = 0.5, linewidth = 70)
ax.hlines(1295, -0.1, 1.1, color = 'red', alpha = 0.5, linewidth = 70)
ax.hlines(1420, -0.1, 1.1, color = 'darkgreen', alpha = 0.5, linewidth = 25)
ax.hlines(1435, -0.1, 1.1, color = 'darkblue', alpha = 0.5, linewidth = 25)
ax.hlines(1640, -0.1, 1.1, color = 'chocolate', alpha = 0.5, linewidth = 70)

colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'black']

ax.set_xlim(-0.1,1.1)


plt.tight_layout()

## Save
save = False
if save == True:
    save_dir = get_directory('Compiled')
    fig.savefig(save_dir + 'Highlighted Spectrum' + '.svg', format = 'svg')
    plt.close(fig)

    
#%% Compiling avg_particles with 2024-04-10 dataset

# Bringing over avg_particles_compiled from 2024-04-10

# for particle in avg_particles:
#     avg_particles_compiled.append(particle)
# avg_particles_compiled[9], avg_particles_compiled[10] = avg_particles_compiled[10], avg_particles_compiled[9]
    
dark_times_compiled = []
for particle in avg_particles_compiled:
    dark_times_compiled.append(particle.dark_time)
   
    
#%% Plot peak area recovery v. dark_time (just for compiled avg particles)
    
powerseries = avg_particles[0].powerseries

scan = np.arange(0, len(powerseries), 1, dtype = int)   

peaks_list = np.array([0,1,2,3,5,6,7,8])

colors = ['grey', 'purple', 'brown', 'red', 'darkgreen', 'darkblue', 'chocolate', 'black']

fig, axes = plt.subplots(len(peaks_list),1,figsize=[12,30], sharex = True)
fig.suptitle('633 nm Powerswitch Recovery - 1 $\mu$W / 90 $\mu$W', fontsize = 'x-large')
axes[len(axes)-1].set_xlabel('Dark time (s)', size = 'x-large')

for i, peak in enumerate(powerseries[0].peaks):
    
    if i in peaks_list:

        
        ## Plot each peak recovery on own axis
        ax = axes[np.where(peaks_list == i)[0][0]]    
        ax.set_xticks(np.linspace(dark_times[0], dark_times[-1], 11))
        ax.set_ylabel('I$_{SERS}$ (%$\Delta$)', size = 'x-large')
              
        ## Get peak recovery v. scan
        y = []
        for particle in avg_particles_compiled:
                y.append(particle.powerseries[0].peaks[i].recovery)            
        y = np.array(y)
        peak_spec = SERS.SERS_Spectrum(x = np.array(dark_times_compiled), y = y)
        color = colors[np.where(peaks_list == i)[0][0]]
        ax.plot(peak_spec.x, peak_spec.y, label = peak.name + 'cm $^{-1}$', color = color, zorder = 1)
        ax.scatter(peak_spec.x, peak_spec.y, marker = 'o', facecolors = 'none', edgecolors = color, linewidth = 3, s = 150, zorder = 2)

        ## Errorbars           
        y_error = []
        for particle in avg_particles_compiled:
            this_error = (particle.powerseries[0].peaks[i].recovery_error)
                
            y_error.append(this_error)   
        y_error = np.array(y_error)
        ax.errorbar(peak_spec.x, peak_spec.y, yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)

        ax.legend(loc = 'upper right')
    
# ax.set_xlim(-100, 1000)
ax.set_xscale('symlog')
axes[len(axes)-1].set_ylabel('I$_{1435}$ / I$_{1420}$', size = 'x-large')
plt.tight_layout(pad = 1.5)
    
## Save
save = False
if save == True:
    save_dir = get_directory('Compiled with 2024-04-10')
    fig.savefig(save_dir + 'Peak Area Recovery' + '.svg', format = 'svg')
    plt.close(fig)


#%% Plot baseline sum recovery v. dark_time (just for compiled avg particles)
    
powerseries = avg_particles[0].powerseries

scan = np.arange(0, len(powerseries), 1, dtype = int)   

fig, ax = plt.subplots(1, 1, figsize=[18,8], sharex = True)
fig.suptitle('633 nm Powerswitch Background Recovery - 1 $\mu$W / 90 $\mu$W', fontsize = 'x-large')
ax.set_xlabel('Dark time (s)', size = 'large')       
ax.set_xticks(np.linspace(dark_times[0], dark_times[-1], 11))
ax.set_ylabel('I$_{Background}$ (%$\Delta$)', size = 'x-large')
      
## Get background sum recovery v. scan
y = []
for particle in avg_particles_compiled:
        y.append(particle.powerseries[0].baseline_sum_recovery)            
y = np.array(y)
peak_spec = SERS.SERS_Spectrum(x = np.array(dark_times_compiled), y = y)
color = 'black'
ax.plot(peak_spec.x, peak_spec.y, color = color, zorder = 1)
ax.scatter(peak_spec.x, peak_spec.y, marker = 'o', facecolors = 'none', edgecolors = color, linewidth = 3, s = 150, zorder = 2)

## Errorbars           
y_error = []
for particle in avg_particles_compiled:
    this_error = (particle.powerseries[0].baseline_sum_recovery_error)
        
    y_error.append(this_error)   
y_error = np.array(y_error)
ax.errorbar(peak_spec.x, peak_spec.y, yerr = y_error, marker = 'none', mfc = color, mec = color, linewidth = 0, markersize = 10, capsize = 7, elinewidth = 3, capthick = 2, ecolor = color, zorder = 1)

ax.legend(loc = 'upper right')
ax.set_xscale('symlog')
plt.tight_layout(pad = 0.5)
    
## Save
save = False
if save == True:
    save_dir = get_directory('Compiled with 2024-04-10')
    fig.savefig(save_dir + 'Background Sum Recovery' + '.svg', format = 'svg')
    plt.close(fig)
