# -*- coding: utf-8 -*-


# Created on Wed Aug 14 15:30:15 2024

# @author: il322

# Plotter for NKT Spot Spread Measurements

# Copied from 2024-12-19_Kymera_Shutter_Si_Raman_Plotter.py

# Data: 
#     "C:\Users\il322\Desktop\Offline Data\2025-01-14_NKT_Spot_Spread_Test.h5"

    




from copy import deepcopy
import gc
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvas
from PIL import Image
from pybaselines.polynomial import modpoly
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
from nplab import ArrayWithAttrs
from lmfit.models import GaussianModel


save = False
save_to_h5 = False


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

## Raw data h5 Files
my_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\2025-01-14_NKT_Spot_Spread_Test.h5")

## Calibration h5 File
cal_h5 = h5py.File(r"C:\Users\il322\Desktop\Offline Data\calibration.h5")

## h5 File for saving processed data
if save_to_h5 == True:
    save_h5 = datafile.DataFile(r"C:\Users\il322\Desktop\Offline Data\2025-01-14_NKT_Spot_Spread_Data.h5")


#%% Quick function to get directory based on particle scan & particle number (one folder per particle) or make one if it doesn't exist

def get_directory(particle_name):
        
    directory_path = r'C:\Users\il322\Desktop\Offline Data\2025-01-14 NKT Spot Spread Analysis\\' + particle_name + '\\'
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    return directory_path


#%% Function to add & process SERS timescan, DF, & CA for each particle


def process_powerswitch_recovery(particle):
    
    # Get powerseries

    keys = list(my_h5[particle.scan][particle.particle_number].keys())
    keys = natsort.natsorted(keys)
    powerseries = []
    for key in keys:
        if 'SERS' in key:
            powerseries.append(my_h5[particle.scan][particle.particle_number][key])

    # Process

    for i, spectrum in enumerate(powerseries):

        spectrum = SERS.SERS_Spectrum(spectrum)
        spectrum.__dict__.pop('dset')

        ## Calibrate x-axis
        spectrum.x = linear(spectrum.x, *cal_matrix)
        spectrum.x = spt.wl_to_wn(spectrum.x, 632.8)
        spectrum.truncate(truncate_range[0], truncate_range[1])
        
        ## Calibrate intensity
        spectrum.calibrate_intensity(R_setup = R_setup,
                                      dark_counts = dark_powerseries[i].y_smooth,
                                      exposure = spectrum.cycle_time,
                                      laser_power = spectrum.laser_power)
        
        ## Modpoly fit for PL background    
        pl = modpoly(spectrum.y, x_data = spectrum.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
        spectrum.pl = pl
        ### Sum of PL background from 350 to 1900 1/cm
        spectrum.pl_sum = np.sum(pl[np.argmin(np.abs(spectrum.x - 350)) : np.argmin(np.abs(spectrum.x - 1900))])
        
        ## Cosmic ray & ALS Baseline Subtraction
        spectrum.y = spt.remove_cosmic_rays(spectrum.y, threshold = 9)
        spectrum.baseline = spt.baseline_als(spectrum.y, 1e3, 1e-2, niter = 10)
        spectrum.y_baselined = spectrum.y - spectrum.baseline
        spectrum.normalise(norm_y = spectrum.y_baselined)
        
        spectrum.peaks = []
        
        powerseries[i] = spectrum

    particle.powerseries = powerseries
    particle.dark_time = np.round(my_h5[particle.scan][particle.particle_number]['dark_time_0'], 0)


def process_df(particle, plot = False, tinder = False, save = False):
    
    ## Get z-scan and images
    keys = list(my_h5[particle.scan][particle.particle_number].keys())
    keys = natsort.natsorted(keys)
    z_scans = []
    images = []
    for key in keys:
        if 'z_scan' in key:
            z_scans.append(my_h5[particle.scan][particle.particle_number][key])
        if 'image' in key:
            images.append(my_h5[particle.scan][particle.particle_number][key])

    ## Process DF
    for i, scan in enumerate(z_scans):
        
        name = str(copy(scan.name))
        
        scan = df.Z_Scan(scan, z_min = scan.attrs['neg'], z_max = scan.attrs['pos'], z_trim = [8, -8])
        scan.truncate(400, 900)   
        scan.condense_z_scan()
        spectrum = scan.df_spectrum
        scan.df_spectrum = df.DF_Spectrum(scan.x, spectrum)
        scan.df_spectrum.find_critical_wln()
        image = images[i]
        save_file = None
        if save == True:
            save_file = get_directory(particle.name) + 'DF_' + str(i) + '.svg'
        scan.df_spectrum = df.df_screening(scan, scan.df_spectrum, image, plot = plot, tinder = tinder, brightness_threshold = 3, title = particle.name + '\n' + str(i), save_file = save_file)
        z_scans[i] = scan
        images[i] = image
        
    ## Add df objects to particle
    particle.z_scans = z_scans
    particle.images = images
    particle.crit_wlns = [z_scans[0].df_spectrum.crit_wln, z_scans[1].df_spectrum.crit_wln]
    particle.is_npom = [particle.z_scans[0].df_spectrum.is_npom, particle.z_scans[0].df_spectrum.is_npom]    


def process_timescan(particle, chunk_size = 20):
    
    ## Get timescan object
    keys = list(my_h5[particle.scan][particle.particle_number].keys())
    keys = natsort.natsorted(keys)
    for key in keys:
        if 'SERS' in key:
            timescan = (my_h5[particle.scan][particle.particle_number][key])
            name = key
    
    ## Timescan calibration
    timescan = SERS.SERS_Timescan(timescan, exposure = timescan.attrs['cycle_time'])
    timescan.x = linear(timescan.x, *cal_matrix)
    timescan.x = spt.wl_to_wn(timescan.x, 632.8)
    timescan.truncate(truncate_range[0], truncate_range[1])
    timescan.calibrate_intensity(R_setup = R_setup,
                                 dark_counts = dark_counts.y_smooth)
    timescan.name = name
    
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
    timescan.__dict__.pop('dset')
    particle.timescan = timescan
    
    
def process_ca(particle):
    
    ca = my_h5[particle.scan][particle.particle_number]['Co-TAPP-SMe_CA_Switch_x5_0']
    ca_y = np.array(ca)[1]
    ca_v = ca.attrs['Levels_v (V)']
    ca = SERS.SERS_Spectrum(x = ca.attrs['Time (s)'], y = ca_y*1e6, particle_name = particle.name)
    ca.name = 'Co-TAPP-SMe_CA_Switch_x5_0'
    ca.v = ca_v
    particle.ca = ca  
    

def process_kinetic_powerseries(particle):
    
    # Add each timescan to powerseries

    ## Get timescan object
    keys = list(my_h5[particle.scan][particle.particle_number].keys())
    keys = natsort.natsorted(keys)
    powerseries = []
    for key in keys:
        if 'SERS' in key:
            powerseries.append(my_h5[particle.scan][particle.particle_number][key])


    # Process each timescan in powerseries

    for i, timescan in enumerate(powerseries):

        name = str(copy(timescan.name))
        timescan = SERS.SERS_Timescan(timescan)
        timescan.__dict__.pop('dset')
        timescan.name = name[name.rfind('/') + 1:]

        ## Calibrate x-axis
        timescan.x = linear(timescan.x, *cal_matrix)
        timescan.x = spt.wl_to_wn(timescan.x, 632.8)
        timescan.truncate(truncate_range[0], truncate_range[1])
        
        ## Calibrate intensity
        timescan.calibrate_intensity(R_setup = R_setup,
                                      dark_counts = dark_powerseries[i].y_smooth,
                                      exposure = timescan.cycle_time,
                                      laser_power = timescan.laser_power)
        
        ## Modpoly fit for PL background    
        # pl = modpoly(timescan.y, x_data = timescan.x, poly_order = 9, max_iter = 10000, tol = 0.001, mask_initial_peaks = True)[0]
        # timescan.pl = pl
        ### Sum of PL background from 350 to 1900 1/cm
        # timescan.pl_sum = np.sum(pl[np.argmin(np.abs(timescan.x - 350)) : np.argmin(np.abs(timescan.x - 1900))])
        
        ## Cosmic ray & ALS Baseline Subtraction
        # timescan.y = spt.remove_cosmic_rays(timescan.y, threshold = 9)
        # timescan.baseline = spt.baseline_als(timescan.y, 1e3, 1e-2, niter = 10)
        # timescan.y_baselined = timescan.y
        # timescan.normalise()
        
        ## Extract nanocavity timescan (using min_spectrum)
        timescan_y = timescan.Y.copy()
        pixel_baseline = np.zeros(len(timescan.x))
        pixel_min = np.zeros(len(timescan.x))
        pixel_sem = np.zeros(len(timescan.x))
        for pixel in range(0,len(timescan.x)):
            pixel_baseline[pixel] = np.polyfit(timescan.t, timescan_y[:,pixel], deg = 0)
            pixel_min[pixel] = np.min(timescan_y[:,pixel])
            pixel_sem[pixel] = np.std(timescan_y[:,pixel])/(len(timescan_y[:,pixel]) ** 0.5)
        timescan.nanocavity_spectrum = pixel_baseline
        timescan.min_spectrum = pixel_min
        timescan.sem_spectrum = pixel_sem
        
        powerseries[i] = timescan
        
    particle.powerseries = powerseries
    particle.chunk_size = 1


#%% Fitting functions

def linear(x, m, b):
    return (m*x) + b


def quadratic(x, a, b, c):
    return (a*x**2 + b*x +c)


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
    

def lorentzian(x, height, mu, fwhm, baseline):
    I = height
    x0 = mu
    gamma = fwhm/2
    numerator = gamma**2
    denominator = (x - x0)**2 + gamma**2
    quot = numerator/denominator
    
    y = I*quot + baseline
    return y


def fit_gaussian(spectrum, fit_range, peak_name = None, R_sq_thresh = 0.9, smooth_first = False, plot = False):    
    
    ## Get region of spectrum to fit
    fit_range_index = [np.abs(spectrum.x-fit_range[0]).argmin(), np.abs(spectrum.x-fit_range[1]).argmin()+1]
    x = spectrum.x[fit_range_index[0]:fit_range_index[1]]   
    
    ## Smooth, truncate y region
    y = spectrum.y_baselined
    if smooth_first == True:
        y = spt.butter_lowpass_filt_filt(y, cutoff = 2000, fs = 40000)
    y = y[fit_range_index[0]:fit_range_index[1]]
    y_raw = spectrum.y_baselined[fit_range_index[0]:fit_range_index[1]]
    

    # Fit
    
    ## Initial guesses
    i_max = y.argmax() # index of highest value - for guess, assumed to be Gaussian peak
    height0 = y[i_max]
    # print(height0)
    
    mu0 = x[i_max] # centre x position of guessed peak
    # print(mu0)
    
    baseline0 = (y[0] + y[len(y)-1])/2 # height of baseline guess
    # print(baseline0)
    
    i_half = np.argmax(y >= (height0 + baseline0)/2) # Index of first argument to be at least halfway up the estimated bell
    if i_half == i_max:
        i_half -= 1
    # Guess sigma from the coordinates at i_half. This will work even if the point isn't at exactly
    # half, and even if this point is a distant outlier the fit should still converge.
    sigma0 = (mu0 - x[i_half]) / np.sqrt(2*np.log((height0 - baseline0)/(y[i_half] - baseline0)))
    width0 = sigma0 * 2 * np.sqrt(2*np.log(2))    
    p0 = height0, mu0, width0, baseline0
    
    ## Perform fit (height, mu, width, baseline)
    start = time.time()
    
    error = False
        
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
        error = True
        print('\nFit Error')
      
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
    if R_sq < R_sq_thresh or np.isnan(R_sq) or error == True:
        # print('\nPoor Fit')
        # print(particle.name)
        # print(spectrum.name)
        # print(peak_name)
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
       
    
    # Plot
    
    if plot == True: # or error == True:
        
        fig, ax = plt.subplots(1,1,figsize=[14,9])
        fig.suptitle(particle.name, fontsize = 'large')
        ax.set_title(spectrum.name + ' - ' + this_peak.name + ' cm$^{-1}$ Peak Fit', fontsize = 'large')
        ax.set_xlabel('Raman Shifts (cm$^{-1}$)')
        ax.set_ylabel('SERS Intensity (cts/mW/s)')
        if smooth_first:
            ax.plot(x, y, color = 'grey', label = 'Smoothed') 
        ax.plot(x, y_raw, color = 'grey', linestyle = '--', label = 'Data')
        ax.plot(x, y_fit, label = 'Fit', color = 'orange')
        ax.plot(x, residuals, label = 'Residuals: ' + str(np.round(residuals_sum, 2)), color = 'black')
        ax.plot(1,1, label = 'R$^2$: ' + str(np.round(R_sq, 3)), color = (0,0,0,0))
        ax.plot(1,1, label = 'Run time (ms): ' + str(np.round(runtime * 1000, 3)), color = (0,0,0,0))
        if error:
            ax.text(x = mu0, y = height0, s = 'ERROR')
        ax.set_xlim(fit_range[0] - 3, fit_range[1] + 3)
        ax.set_ylim(None, y.max()*1.2)
        ax.legend()
        plt.show()
        
    return this_peak

# peak = fit_gaussian(spectrum = spectrum, fit_range = [480, 525], plot = True, smooth_first = False, R_sq_thresh=0.1)


#%% Run to here then load in spydata after processing

'''
Run up to here then load in spydata after processing
'''


'''
Running data processing below
'''


#%% Plot spectrum v. position for each lens at each wavelength


lenses = [my_h5['AC254-050-A_0'], my_h5['no_lens_1']]

for lens in lenses:

    wavelengths = np.arange(450, 860, 50)

    for wavelength in wavelengths:
        
        spectra = []
        ls = []
        
        groups = list(lens.keys())
        groups = natsort.natsorted(groups)
        for group in groups:
            if group[-2:] != 'mm':
                continue
        
            l = float(group[0:group.find('mm')])
            # print(l)
            group = lens[group]    
            keys = list(group.keys())
            keys = natsort.natsorted(keys)
            
            for key in keys:
                if str(wavelength)+'nm' in key:
                    spectra.append(group[key])
                    ls.append(l)
                    
        ys = []           
        for i, spectrum in enumerate(spectra):
            
            # spectrum = group
            x = spectrum.attrs['wavelengths']
            y = np.array(spectrum)
            background = spectrum.attrs['background']
            spectrum = SERS.SERS_Spectrum(spectrum)
            spectrum.background = background
            spectrum.l = ls[i]
            spectrum.name = spectrum.name + ' @ ' + str(l) + 'mm'
            spectrum.y = spectrum.y - spectrum.background
            spectra[i] = spectrum
            ys.append(y)
        ys = np.array(ys)
            
        fig, ax = plt.subplots(1,1,figsize=[12,9])
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.set_title('NKT Spot Spread @ ' + str(wavelength) + ' nm\n' + lens.name[1:])
        
        ## Color scale & colorbar
        my_cmap = plt.get_cmap('plasma')
        cmap = my_cmap
        norm = mpl.colors.Normalize(vmin = ls[0], vmax = ls[-1])
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        cbar_ax.set_ylabel('Spot Position (mm)', rotation=270, labelpad = 20)
        cbar_ax.yaxis.set_label_position("right")
        # cbar_ax.set_yticks(np.geomspace(0.01, 1, 5), np.geomspace(0.01, 1, 5))
        cbar_ax.tick_params(which = 'major', length = 10, width = 1.5, labelsize = 'small')
        cbar_ax.tick_params(which = 'minor', length = 4, width = 1)
        
        offset = 0
        for spectrum in spectra:
            color = cmap(norm(spectrum.l))
            ax.plot(spectrum.x, spectrum.y + (spectrum.l*offset), color = color, alpha = 0.5)
            
        ax.set_yscale('log')    
        
        ## Save
        if save == True:
            save_dir = get_directory(lens.name[1:])
            print('yes')
            fig.savefig(save_dir + 'NKT Spot Spread ' + str(wavelength) + ' nm' + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)
                
    
#%% Timescan - Plot spectrum v. position for each lens at each wavelength


lenses = [my_h5['AC254-050-A_0'], my_h5['no_lens_1']]

for lens in lenses:
    
    wavelengths = np.arange(450, 860, 50)

    for wavelength in wavelengths:
        
        spectra = []
        ls = []
        
        groups = list(lens.keys())
        groups = natsort.natsorted(groups)
        for group in groups:
            if group[-2:] != 'mm':
                continue
        
            l = float(group[0:group.find('mm')])
            # print(l)
            group = lens[group]    
            keys = list(group.keys())
            keys = natsort.natsorted(keys)
            
            for key in keys:
                if str(wavelength)+'nm' in key:
                    spectra.append(group[key])
                    ls.append(l)
                    
        ys = []           
        for i, spectrum in enumerate(spectra):
            
            # spectrum = group
            x = spectrum.attrs['wavelengths']
            y = np.array(spectrum)
            background = spectrum.attrs['background']
            spectrum = SERS.SERS_Spectrum(spectrum)
            spectrum.background = background
            spectrum.l = ls[i]
            spectrum.name = spectrum.name + ' @ ' + str(l) + 'mm'
            spectrum.y = spectrum.y - spectrum.background
            spectra[i] = spectrum
            ys.append(y)
        ys = np.array(ys)
            
        fig, ax = plt.subplots(1,1,figsize=[12,9])
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Spot Position (mm)')
        ax.set_title('NKT Spot Spread @ ' + str(wavelength) + ' nm\n' + lens.name[1:])
        
        ## Color scale & colorbar
        # my_cmap = plt.get_cmap('plasma')
        # cmap = my_cmap
        # norm = mpl.colors.Normalize(vmin = 0, vmax = 10)
        # cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        # fig.colorbar(mappable = cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        # cbar_ax.set_ylabel('Spot Position (mm)', rotation=270, labelpad = 20)
        # cbar_ax.yaxis.set_label_position("right")
        # # cbar_ax.set_yticks(np.geomspace(0.01, 1, 5), np.geomspace(0.01, 1, 5))
        # cbar_ax.tick_params(which = 'major', length = 10, width = 1.5, labelsize = 'small')
        # cbar_ax.tick_params(which = 'minor', length = 4, width = 1)
        
        # offset = 0
        # for spectrum in spectra:
        #     color = cmap(norm(spectrum.l))
        #     ax.plot(spectrum.x, spectrum.y + (spectrum.l*offset), color = color, alpha = 0.5)
            
        
        # ax.set_title('633 nm 1 $\mu$W SERS Timescan', fontsize = 'x-large', pad = 20)
        pcm = ax.pcolormesh(spectrum.x, ls, ys, vmin = 0, vmax = ys.max(), cmap = cmap, rasterized = 'True')
        clb = fig.colorbar(pcm, ax=ax)
        clb.set_label(label = 'Intensity (Counts)', size = 'x-large', rotation = 270, labelpad=30)
        clb.ax.tick_params(labelsize='large')
        
        # ax.set_yscale('log')    
                
        ## Save
        if save == True:
            save_dir = get_directory(lens.name[1:])
            fig.savefig(save_dir + '2D NKT Spot Spread ' + str(wavelength) + ' nm' + '.svg', format = 'svg', bbox_inches='tight')
            plt.close(fig)
            
            
#%% Plot power spectrum for each of the lenses tried

lenses = [my_h5['AC254-050-A_0']['power_calibration_1'], 
          my_h5['no_lens_1']['power_calibration_0'], 
          my_h5['AC254-050-AB_0']['power_calibration_0']]

fig, ax = plt.subplots(1,1,figsize=[12,9])
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Power (mW)')
ax.set_title('Power spectrum v. Focusing lens')


for group in lenses:

    lens = group.parent.name[1:]

    # Get power spectrum
    
    power_dict = {} 

    for key in group.keys():

        power = group[key]
        wavelength = int(power.attrs['wavelength'])
        power = float(np.array(power))
        new_dict = {wavelength : power}
        power_dict.update(new_dict)

    group.power_dict = power_dict
    
    
    # Plot
                
    ax.plot(group.power_dict.keys(), group.power_dict.values(), label = lens)

ax.legend(title = 'Lens')

## Save
if save == True:
    save_dir = get_directory('Compiled')
    fig.savefig(save_dir + 'NKT Output Spectrum v Lens' + '.svg', format = 'svg', bbox_inches='tight')
    plt.close(fig)